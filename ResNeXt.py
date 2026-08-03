import scipy.io as sio  # import scipy.io for .mat file I/O
import numpy as np  # import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import os  # For manipulating folders and acquiring file size
import glob  # For matching video filenames
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# ==========================================
# Core import: PennyLane Variational Quantum Circuit (VQC) library
# ==========================================
import pennylane as qml

# Retain physical layer optimization algorithm from original code
from optimization import bisection

# Set random seeds to ensure quantum and reinforcement learning experiments are reproducible
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1. Variational Quantum Circuit (PQC / VQC) Architecture Standardized to Paper
# ==========================================
n_qubits = 6  # Number of qubits set based on paper's wireless network size (usually 4 ~ 8)
n_layers = 2  # Number of quantum variational circuit layers (Ansatz depth)

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Paper-standard VQC architecture:
    - Inputs: Rotation angles encoded from classical channel state, shape (Batch_size, n_qubits)
    - Weights: Trainable variational parameters, shape (n_layers, n_qubits, 3) (corresponding to RX, RY, RZ)
    """
    # 1. Classical data encoding layer (State Encoding Layer: Angle Encoding)
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)

    # 2. Multi-layer variational & entangling blocks (Variational & Entangling Layers)
    for l in range(n_layers):
        # (a) Variational rotation layer: single-qubit rotation along three axes
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

        # (b) Ring/Full entanglement topology layer (Entanglement Topology)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # 3. Quantum Pauli-Z measurement (Expectation Value Measurement)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==========================================
# 2. 1D ResNeXt Block (based on Cardinality and Grouped Convolution)
# ==========================================
class ResNeXtBlock1D(nn.Module):
    """
    1D ResNeXt bottleneck block (Split-Transform-Merge strategy)
    - Cardinality: Number of parallel grouped transformation paths
    - Bottleneck Width: Channel width of each grouped path
    """
    def __init__(self, in_channels, out_channels, cardinality=4, bottleneck_width=8, stride=1):
        super(ResNeXtBlock1D, self).__init__()
        D = cardinality * bottleneck_width  # Total channels in bottleneck layer

        # 1x1 convolution: channel compression (Split)
        self.conv1 = nn.Conv1d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(D)

        # 3x1 grouped convolution: multi-path independent transformation (Transform via Cardinality)
        self.conv2 = nn.Conv1d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm1d(D)

        # 1x1 convolution: channel expansion/restoration (Merge)
        self.conv3 = nn.Conv1d(D, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut Connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += residual  # Residual fusion
        return self.relu(out)


class ResNeXtEncoder(nn.Module):
    """Classical channel encoder based on ResNeXt architecture"""
    def __init__(self, input_dim, n_qubits, cardinality=4, bottleneck_width=8):
        super(ResNeXtEncoder, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.resnext_block = ResNeXtBlock1D(in_channels=16, out_channels=32, cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.fc = nn.Linear(32 * input_dim, n_qubits)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x_seq = x.unsqueeze(1)
        out = self.input_conv(x_seq)
        out = self.resnext_block(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.tanh(out)


class ResNeXtDecoder(nn.Module):
    """Classical action decoder based on ResNeXt architecture"""
    def __init__(self, n_qubits, output_dim, cardinality=4, bottleneck_width=8):
        super(ResNeXtDecoder, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.resnext_block = ResNeXtBlock1D(in_channels=16, out_channels=32, cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.fc = nn.Linear(32 * n_qubits, output_dim)

    def forward(self, q_features):
        # q_features: (batch_size, n_qubits) -> (batch_size, 1, n_qubits)
        q_seq = q_features.unsqueeze(1)
        out = self.input_conv(q_seq)
        out = self.resnext_block(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==========================================
# 3. QuantumMemoryDNN Integrated with ResNeXt (ResNeXt-HQNN)
# ==========================================
class QuantumMemoryDNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network (ResNeXt-HQNN) memory decoding architecture integrated with ResNeXt
    - Classical Encoder: Extracts multi-path features of multi-channel gain h based on ResNeXt grouped convolutions
    - Quantum Layer: PennyLane Variational Quantum Circuit
    - Classical Decoder: Reconstructs continuous actions and Q-values based on ResNeXt multi-path structure
    """

    def __init__(self, net_dims, learning_rate=0.01, training_interval=10, batch_size=128, memory_size=1024):
        super(QuantumMemoryDNN, self).__init__()
        self.input_dim = net_dims[0]  # Number of users N (channel gain dimension)
        self.output_dim = net_dims[-1]  # Output dimension N (corresponding to binary offloading/mode vector)
        self.n_qubits = n_qubits

        # (1) Classical encoding layer based on ResNeXt
        self.encoder = ResNeXtEncoder(input_dim=self.input_dim, n_qubits=self.n_qubits, cardinality=4, bottleneck_width=8)

        # (2) PennyLane variational quantum layer (Quantum Layer)
        weight_shapes = {"weights": (n_layers, self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # (3) Classical decoding layer based on ResNeXt
        self.decoder = ResNeXtDecoder(n_qubits=self.n_qubits, output_dim=self.output_dim, cardinality=4, bottleneck_width=8)

        # Components required for reinforcement learning and training
        self.learning_rate = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=self.memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.cost_his = []
        self.time_step = 0

    def forward(self, x):
        # 1. Classical ResNeXt encoding: extract multi-branch features and scale to radian range [-pi, pi]
        x_angles = self.encoder(x) * np.pi

        # 2. Quantum circuit computation & expectation value measurement
        quantum_features = self.q_layer(x_angles)

        # 3. Classical ResNeXt decoding: reconstruct and output Q-values / continuous action space
        q_values = self.decoder(quantum_features)
        return q_values

    def decode(self, h, K, mode='OP'):
        """Generate K candidate action modes according to order-preserving quantization in the paper"""
        self.eval()  # In evaluation mode, BatchNorm1d uses global statistics
        h_tensor = torch.FloatTensor(h).unsqueeze(0)  # Convert to (1, N)
        with torch.no_grad():
            q_values = self.forward(h_tensor).squeeze(0).cpu().numpy()

        if mode == 'OP':
            return self._order_preserving_quantization(q_values, K)
        else:
            return self._knn_quantization(q_values, K)

    def _order_preserving_quantization(self, q_values, K):
        """Core algorithm in paper: order-preserving quantization to generate decision set"""
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
        # Quantum gradient clipping to prevent barren plateaus in variational quantum circuits
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.cost_his.append(loss.item())

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_his, color='purple', linewidth=1.5)
        plt.ylabel('ResNeXt-Quantum Loss')
        plt.xlabel('Training Iterations')
        plt.title('Convergence Curve of ResNeXt-VQC LightQDRL')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


# ==========================================
# 4. Data Saving and Plotting Utility Functions
# ==========================================
def save_data(data, file_path):
    """Save numerical value list to specified text path"""
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
    plt.title('Normalized Computation Rate vs. Time Frames (ResNeXt-LightQDRL)')
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
    plt.title('System Throughput over Time Frames (ResNeXt-LightQDRL)')
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
    plt.plot(sorted_latency, cdf, color='#d62728', linewidth=2.0, label='ResNeXt-LightQDRL Latency CDF')

    plt.ylabel('Cumulative Probability (CDF)')
    plt.xlabel('Transmission & Processing Latency (ms)')
    plt.title('Cumulative Distribution Function (CDF) of Latency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def calculate_video_transmission_time(task_folder, avg_normalized_rate, results_dir="./result-ResNeXt"):
    """5G real physical transmission latency evaluation module"""
    REAL_5G_BASE_RATE_MBPS = 300.0
    actual_avg_rate_bps = avg_normalized_rate * REAL_5G_BASE_RATE_MBPS * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (5G Network Standard) ---")
    print(f"Quantum ResNeXt network offline normalized efficiency: {avg_normalized_rate:.4f}")
    print(f"Current predicted real 5G average transmission rate: {actual_avg_rate_bps / 1e6:.3f} Mbps")

    search_path = os.path.join(task_folder, "*.mp4")
    video_files = glob.glob(search_path)

    if not video_files:
        print(f"Notice: No .mp4 video files detected in '{task_folder}' folder.")
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
        f.write("    PennyLane ResNeXt-integrated Quantum Reinforcement Learning (ResNeXt-LightQDRL) Report\n")
        f.write("=========================================================\n")
        f.write(f"5G Network Scenario Baseline Setting: {REAL_5G_BASE_RATE_MBPS:.1f} Mbps\n")
        f.write(f"Quantum Variational Network Optimized Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps\n")
        f.write(f"Total Processed Video Files: {len(video_files)}\n")
        f.write(f"Total Video File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} seconds\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"{'Video Filename':<30} | {'File Size (MB)':<15} | {'Est. Transmission Time (s)':<15}\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")
        f.write("=========================================================\n")

    print(f"Video transmission time analysis completed! Real quantum environment report exported to: '{output_file}'\n")


# ==========================================
# 5. Main Program Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Create results storage folder result-ResNeXt
    results_dir = "./result-ResNeXt"
    os.makedirs(results_dir, exist_ok=True)

    N = 10  # Number of users/channels
    n = 30000  # Total number of time frames
    K = N  # Initialize candidate solution space dimension K
    decoder_mode = 'OP'  # Quantization decoding mode ('OP' is Order-Preserving recommended in paper)
    Memory = 1024  # Experience replay memory capacity
    Delta = 32  # Adaptive K adjustment interval
    REAL_5G_BASE_RATE_MBPS = 300.0  # Baseline 5G bandwidth rate (Mbps)
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # Assumed edge task size (5 MB = 40 Mbits)

    print(
        '#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (N, n, K, decoder_mode, Memory, Delta))

    # 1. Read channel data and theoretical baseline rate
    channel = sio.loadmat('./data/data_%d' % N)['input_h']
    rate = sio.loadmat('./data/data_%d' % N)['output_obj']

    channel = channel * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    # 2. Instantiate quantum variational neural network integrated with ResNeXt (ResNeXt-LightQDRL)
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

    # 3. Real-time online optimization loop (Online Optimization Loop)
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

        # (a) Call ResNeXt + VQC decoding layer to generate candidate mode order-preserving list
        m_list = mem.decode(h, K, decoder_mode)

        # (b) Physical layer bisection method optimization (Bisection Method Optimization)
        r_list = []
        for m in m_list:
            r_list.append(bisection(h / 1000000, m)[0])

        # (c) Store optimal decision and trigger ResNeXt-VQC backpropagation update
        best_rate = np.max(r_list)
        mem.encode(h, m_list[np.argmax(r_list)])

        # Core metric calculations:
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

    # 4. Export data to ./result-ResNeXt/ folder
    save_data(rate_his_ratio, os.path.join(results_dir, "normalized_computation_rate.txt"))
    save_data(throughput_his, os.path.join(results_dir, "throughput_mbps.txt"))
    save_data(latency_his, os.path.join(results_dir, "latency_ms.txt"))
    save_data(k_idx_his, os.path.join(results_dir, "k_idx_his.txt"))
    save_data(K_his, os.path.join(results_dir, "K_his.txt"))
    save_data(mem.cost_his, os.path.join(results_dir, "cost_his.txt"))

    np.save(os.path.join(results_dir, "normalized_computation_rate.npy"), np.array(rate_his_ratio))
    np.save(os.path.join(results_dir, "throughput_mbps.npy"), np.array(throughput_his))
    np.save(os.path.join(results_dir, "latency_ms.npy"), np.array(latency_his))

    # 5. Plot and save three performance metrics and convergence curve independently to ./result-ResNeXt/
    mem.plot_cost(save_path=os.path.join(results_dir, "quantum_loss_convergence.png"))
    plot_normalized_rate(rate_his_ratio, save_path=os.path.join(results_dir, "normalized_computation_rate.png"))
    plot_throughput(throughput_his, save_path=os.path.join(results_dir, "throughput.png"))
    plot_cdf_latency(latency_his, save_path=os.path.join(results_dir, "cdf_of_latency.png"))

    # 6. 5G video transmission performance evaluation test
    calculate_video_transmission_time(task_folder="./task",
                                      avg_normalized_rate=avg_normalized_rate,
                                      results_dir=results_dir)

    # rate: 0.9939282386501899
    # Total time  consumed: 603.0539939403534
    # Average time  per channel: 0.020101799798011778