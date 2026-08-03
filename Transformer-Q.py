import scipy.io as sio  # import scipy.io for .mat file I/O
import numpy as np  # import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import os  # Used for directory manipulation and getting file sizes
import glob  # Used for matching video file names
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import math

# ==========================================
# Core Import: PennyLane Variational Quantum Circuit (VQC) Library
# ==========================================
import pennylane as qml

# Retain physical layer optimization algorithm from original code
from optimization import bisection

# Set random seeds for reproducibility of quantum and RL experiments
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1. Paper-Standard Variational Quantum Circuit (PQC / VQC) Architecture
# ==========================================
n_qubits = 6  # Number of qubits set based on wireless network size in paper (typically 4 ~ 8)
n_layers = 2  # Number of quantum variational circuit layers (Ansatz depth)

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Paper-standard VQC Architecture:
    - Inputs: Rotation angles encoded from classical channel state, shape (Batch_size, n_qubits)
    - Weights: Trainable variational parameters, shape (n_layers, n_qubits, 3) (corresponding to RX, RY, RZ)
    """
    # 1. Classical Data Encoding Layer (State Encoding Layer: Angle Encoding)
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)

    # 2. Multi-layer Variational & Entangling Blocks
    for l in range(n_layers):
        # (a) Variational rotation layer: single-qubit rotations on three axes
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

        # (b) Ring/Full entanglement layer (Entanglement Topology)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # 3. Quantum Pauli-Z Measurement (Expectation Value Measurement)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==========================================
# 2. Transformer-based Classical Encoder & Decoder Modules
# ==========================================
class PositionalEncoding(nn.Module):
    """Positional Encoding to inject positional/spatial information into sequential inputs"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """
    Transformer-based classical channel encoder
    Uses Multi-Head Self-Attention to capture cross-correlations among multi-user channels
    Maps (batch_size, input_dim) -> (batch_size, n_qubits) angle space
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, n_qubits=6):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim * d_model, n_qubits)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        batch_size = x.size(0)
        x_seq = x.unsqueeze(-1)
        out = self.embedding(x_seq)  # (batch_size, input_dim, d_model)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)  # (batch_size, input_dim, d_model)
        out = out.reshape(batch_size, -1)  # Flatten feature vectors
        out = self.fc_out(out)  # (batch_size, n_qubits)
        return self.tanh(out)


class TransformerDecoder(nn.Module):
    """
    Transformer-based classical action decoder
    Uses Self-Attention to deconstruct quantum features and reconstruct continuous action Q-values
    Maps (batch_size, n_qubits) -> (batch_size, output_dim)
    """
    def __init__(self, n_qubits, d_model=64, nhead=4, num_layers=2, output_dim=10):
        super(TransformerDecoder, self).__init__()
        self.n_qubits = n_qubits
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(n_qubits * d_model, output_dim)

    def forward(self, q_features):
        # q_features: (batch_size, n_qubits) -> (batch_size, n_qubits, 1)
        batch_size = q_features.size(0)
        q_seq = q_features.unsqueeze(-1)
        out = self.embedding(q_seq)  # (batch_size, n_qubits, d_model)
        out = self.pos_encoder(out)
        out = self.transformer_decoder(out)  # (batch_size, n_qubits, d_model)
        out = out.reshape(batch_size, -1)  # Flatten feature vectors
        out = self.fc_out(out)  # (batch_size, output_dim)
        return out


# ==========================================
# 3. Transformer-integrated QuantumMemoryDNN (Transformer-HQNN)
# ==========================================
class QuantumMemoryDNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network (Transformer-HQNN) memory decoding architecture integrating Transformer
    - Classical Encoder: Uses Transformer multi-head self-attention to extract global dependencies of multi-channel gains h
    - Quantum Layer: PennyLane Variational Quantum Circuit
    - Classical Decoder: Uses Transformer self-attention to reconstruct continuous actions and Q-values
    """

    def __init__(self, net_dims, learning_rate=0.01, training_interval=10, batch_size=128, memory_size=1024):
        super(QuantumMemoryDNN, self).__init__()
        self.input_dim = net_dims[0]  # Number of users N (channel gain dimension)
        self.output_dim = net_dims[-1]  # Output dimension N (corresponding to binary offloading/mode vector)
        self.n_qubits = n_qubits

        # (1) Classical Transformer encoding layer
        self.encoder = TransformerEncoder(
            input_dim=self.input_dim, d_model=64, nhead=4, num_layers=2, n_qubits=self.n_qubits
        )

        # (2) PennyLane Variational Quantum Layer
        weight_shapes = {"weights": (n_layers, self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # (3) Classical Transformer decoding layer
        self.decoder = TransformerDecoder(
            n_qubits=self.n_qubits, d_model=64, nhead=4, num_layers=2, output_dim=self.output_dim
        )

        # Components required for Reinforcement Learning and training
        self.learning_rate = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=self.memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.cost_his = []
        self.time_step = 0

    def forward(self, x):
        # 1. Classical Transformer encoding: extract multi-head attention features and scale to radian range [-pi, pi]
        x_angles = self.encoder(x) * np.pi

        # 2. Quantum circuit execution & expectation measurement
        quantum_features = self.q_layer(x_angles)

        # 3. Classical Transformer decoding: reconstruct and output Q-values / continuous action space
        q_values = self.decoder(quantum_features)
        return q_values

    def decode(self, h, K, mode='OP'):
        """Generate K candidate action modes following Order-Preserving quantization as described in the paper"""
        self.eval()
        h_tensor = torch.FloatTensor(h).unsqueeze(0)  # Convert to (1, N)
        with torch.no_grad():
            q_values = self.forward(h_tensor).squeeze(0).cpu().numpy()

        if mode == 'OP':
            return self._order_preserving_quantization(q_values, K)
        else:
            return self._knn_quantization(q_values, K)

    def _order_preserving_quantization(self, q_values, K):
        """Core paper algorithm: Order-Preserving quantization to generate decision set"""
        idx = np.argsort(q_values)
        m_list = np.zeros((K, self.output_dim))
        for k in range(K):
            if k < self.output_dim:
                m_list[k, idx[-(k + 1):]] = 1
        return m_list

    def _knn_quantization(self, q_values, K):
        m_list = []
        base_m = (q_values > 0).astype(int)
        for _ in range(K):
            m = base_m.copy()
            if random.random() < 0.2:
                flip_idx = random.randint(0, self.output_dim - 1)
                m[flip_idx] = 1 - m[flip_idx]
            m_list.append(m)
        return np.array(m_list)

    def encode(self, h, best_m):
        """Push optimal channel-decision pair into Experience Replay Memory"""
        self.memory.append((h, best_m))
        self.time_step += 1
        if self.time_step % self.training_interval == 0 and len(self.memory) >= self.batch_size:
            self._train_step()

    def _train_step(self):
        self.train()
        batch = random.sample(self.memory, self.batch_size)
        h_batch, m_batch = zip(*batch)

        h_tensor = torch.FloatTensor(np.array(h_batch))
        m_tensor = torch.FloatTensor(np.array(m_batch))

        pred_q = self.forward(h_tensor)
        loss = F.binary_cross_entropy_with_logits(pred_q, m_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        # Quantum gradient clipping to prevent Barren Plateaus in variational quantum circuits
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.cost_his.append(loss.item())

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_his, color='purple', linewidth=1.5)
        plt.ylabel('Transformer-Quantum Loss')
        plt.xlabel('Training Iterations')
        plt.title('Convergence Curve of Transformer-VQC LightQDRL')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


# ==========================================
# 4. Data Saving and Plotting Functions
# ==========================================
def save_data(data, file_path):
    """Save numerical list to specified text file path"""
    with open(file_path, 'w') as f:
        for val in data:
            f.write("%s\n" % val)


def plot_normalized_rate(rate_his_ratio, save_path=None, rolling_intv=50):
    """Plot and save 1. Normalized Computation Rate figure"""
    rate_array = np.asarray(rate_his_ratio)
    df = pd.DataFrame(rate_array)

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_curve = np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values)
    min_curve = np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values)
    max_curve = np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values)
    frames = np.arange(len(rate_array)) + 1

    plt.plot(frames, mean_curve, color='#1f77b4', label='Moving Average', linewidth=1.8)
    plt.fill_between(frames, min_curve, max_curve, color='#1f77b4', alpha=0.2, label='Min-Max Bounds')

    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.title('Normalized Computation Rate vs. Time Frames (Transformer-LightQDRL)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_throughput(throughput_his, save_path=None, rolling_intv=50):
    """Plot and save 2. Throughput (Mbps) figure"""
    tp_array = np.asarray(throughput_his)
    df = pd.DataFrame(tp_array)

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_curve = np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values)
    min_curve = np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values)
    max_curve = np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values)
    frames = np.arange(len(tp_array)) + 1

    plt.plot(frames, mean_curve, color='#2ca02c', label='System Throughput', linewidth=1.8)
    plt.fill_between(frames, min_curve, max_curve, color='#2ca02c', alpha=0.2, label='Fluctuation Region')

    plt.ylabel('Throughput (Mbps)')
    plt.xlabel('Time Frames')
    plt.title('System Throughput over Time Frames (Transformer-LightQDRL)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cdf_latency(latency_his, save_path=None):
    """Plot and save 3. CDF of Latency (ms) figure"""
    sorted_latency = np.sort(np.asarray(latency_his))
    cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(sorted_latency, cdf, color='#d62728', linewidth=2.0, label='Transformer-LightQDRL Latency CDF')

    plt.ylabel('Cumulative Probability (CDF)')
    plt.xlabel('Transmission & Processing Latency (ms)')
    plt.title('Cumulative Distribution Function (CDF) of Latency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def calculate_video_transmission_time(task_folder, avg_normalized_rate, results_dir="./result-Transformer"):
    """Real physical 5G transmission latency evaluation module"""
    REAL_5G_BASE_RATE_MBPS = 300.0
    actual_avg_rate_bps = avg_normalized_rate * REAL_5G_BASE_RATE_MBPS * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (5G Network Standard) ---")
    print(f"Quantum Transformer network offline normalized efficiency: {avg_normalized_rate:.4f}")
    print(f"Currently predicted real 5G average transmission rate: {actual_avg_rate_bps / 1e6:.3f} Mbps")

    search_path = os.path.join(task_folder, "*.mp4")
    video_files = glob.glob(search_path)

    if not video_files:
        print(f"Notice: No .mp4 video files detected in folder '{task_folder}'.")
        if not os.path.exists(task_folder):
            os.makedirs(task_folder)
        return

    results = []
    total_size_bytes = 0
    total_time_seconds = 0

    for file_path in video_files:
        file_name = os.path.basename(file_path)
        file_size_bytes = os.path.getsize(file_path)
        file_size_bits = file_size_bytes * 8
        transmission_time = file_size_bits / actual_avg_rate_bps

        total_size_bytes += file_size_bytes
        total_time_seconds += transmission_time

        results.append({
            'name': file_name,
            'size_mb': file_size_bytes / (1024 * 1024),
            'time_sec': transmission_time
        })

    output_file = os.path.join(results_dir, "transmission_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("    PennyLane Transformer-integrated Quantum Reinforcement Learning (Transformer-LightQDRL) Report\n")
        f.write("=========================================================\n")
        f.write(f"5G Network Scenario Benchmark Setting: {REAL_5G_BASE_RATE_MBPS:.1f} Mbps\n")
        f.write(f"Quantum Variational Network Optimized Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps\n")
        f.write(f"Total Video Files Processed: {len(video_files)}\n")
        f.write(f"Total Video File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} seconds\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"{'Video File Name':<30} | {'File Size (MB)':<15} | {'Est. Time (sec)':<15}\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")
        f.write("=========================================================\n")

    print(f"Video transmission time analysis complete! Real quantum environment report exported to: '{output_file}'\n")


# ==========================================
# 5. Main Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Create result storage folder: result-Transformer
    results_dir = "./result-Transformer"
    os.makedirs(results_dir, exist_ok=True)

    N = 10  # Number of users/channels
    n = 30000  # Total number of time frames
    K = N  # Initialize candidate solution space dimension K
    decoder_mode = 'OP'  # Quantization decoding mode ('OP' is Order-Preserving recommended in paper)
    Memory = 1024  # Experience replay memory capacity
    Delta = 32  # Adaptive K adjustment interval
    REAL_5G_BASE_RATE_MBPS = 300.0  # Benchmark 5G bandwidth rate (Mbps)
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # Assumed edge task size (5 MB = 40 Mbits)

    print(
        '#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (N, n, K, decoder_mode, Memory, Delta))

    # 1. Read channel data and theoretical benchmark rate
    channel = sio.loadmat('./data/data_%d' % N)['input_h']
    rate = sio.loadmat('./data/data_%d' % N)['output_obj']

    channel = channel * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    # 2. Instantiate Transformer-integrated Quantum Variational Neural Network (Transformer-LightQDRL)
    mem = QuantumMemoryDNN(net_dims=[N, 64, N],
                           learning_rate=0.01,
                           training_interval=10,
                           batch_size=128,
                           memory_size=Memory
                           )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    throughput_his = []
    latency_his = []
    mode_his = []
    k_idx_his = []
    K_his = []

    # 3. Real-time Online Optimization Loop
    for i in range(n):
        if i % (n // 10) == 0:
            print("%0.1f" % (i / n))
        if i > 0 and i % Delta == 0:
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) + 1
            else:
                max_k = k_idx_his[-1] + 1
            K = min(max_k + 1, N)

        if i < n - num_test:
            i_idx = i % split_idx
        else:
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx, :]

        # (a) Call Transformer + VQC decoding layer to generate candidate mode list via Order-Preserving
        m_list = mem.decode(h, K, decoder_mode)

        # (b) Physical layer bisection method optimization
        r_list = []
        for m in m_list:
            r_list.append(bisection(h / 1000000, m)[0])

        # (c) Store optimal decision and trigger Transformer-VQC backpropagation update
        best_rate = np.max(r_list)
        mem.encode(h, m_list[np.argmax(r_list)])

        # Core metrics calculation:
        # 1. Normalized Computation Rate
        norm_rate = best_rate / rate[i_idx][0]
        # 2. Throughput (Mbps)
        current_throughput_mbps = norm_rate * REAL_5G_BASE_RATE_MBPS
        # 3. Latency (ms): Task_Size / Rate
        current_latency_ms = (TASK_SIZE_BITS / (current_throughput_mbps * 1e6)) * 1000.0

        rate_his.append(best_rate)
        rate_his_ratio.append(norm_rate)
        throughput_his.append(current_throughput_mbps)
        latency_his.append(current_latency_ms)

        k_idx_his.append(np.argmax(r_list))
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])

    total_time = time.time() - start_time

    avg_normalized_rate = sum(rate_his_ratio[-num_test: -1]) / num_test
    print("Averaged normalized computation rate:", avg_normalized_rate)
    print('Total time consumed:%s' % total_time)
    print('Average time per channel:%s' % (total_time / n))

    # 4. Export data to ./result-Transformer/ directory
    save_data(rate_his_ratio, os.path.join(results_dir, "normalized_computation_rate.txt"))
    save_data(throughput_his, os.path.join(results_dir, "throughput_mbps.txt"))
    save_data(latency_his, os.path.join(results_dir, "latency_ms.txt"))
    save_data(k_idx_his, os.path.join(results_dir, "k_idx_his.txt"))
    save_data(K_his, os.path.join(results_dir, "K_his.txt"))
    save_data(mem.cost_his, os.path.join(results_dir, "cost_his.txt"))

    np.save(os.path.join(results_dir, "normalized_computation_rate.npy"), np.array(rate_his_ratio))
    np.save(os.path.join(results_dir, "throughput_mbps.npy"), np.array(throughput_his))
    np.save(os.path.join(results_dir, "latency_ms.npy"), np.array(latency_his))

    # 5. Plot and save three performance metrics and convergence curves to ./result-Transformer/
    mem.plot_cost(save_path=os.path.join(results_dir, "quantum_loss_convergence.png"))
    plot_normalized_rate(rate_his_ratio, save_path=os.path.join(results_dir, "normalized_computation_rate.png"))
    plot_throughput(throughput_his, save_path=os.path.join(results_dir, "throughput.png"))
    plot_cdf_latency(latency_his, save_path=os.path.join(results_dir, "cdf_of_latency.png"))

    # 6. 5G video transmission performance evaluation test
    calculate_video_transmission_time(task_folder="./task",
                                      avg_normalized_rate=avg_normalized_rate,
                                      results_dir=results_dir)

    # rate: 0.9522576739784383
    # Total time  consumed: 843.3704121112823
    # Average time  per channel: 0.02811234707037608