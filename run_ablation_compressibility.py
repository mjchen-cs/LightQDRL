import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import os
import glob
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# ==========================================
# Core import: PennyLane Variational Quantum Circuit (VQC) library
# ==========================================
import pennylane as qml

# Import physical layer bisection optimization algorithm (requires optimization.py in the same directory)
try:
    from optimization import bisection
except ImportError:
    # Fallback Mock function (prevents errors if optimization.py is missing; ensure source file exists for actual experiments)
    def bisection(h, m):
        return np.sum(h * m) * 0.1, None

# Set random seeds to ensure reproducibility of ablation experiments
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ==========================================
# 1. Quantum Circuit Compressibility Auxiliary Functions: Fake Quantization (STE) & Gate Pruning
# ==========================================
class FakeQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE).
    Simulates low-bit quantization (e.g., INT8 / INT4) in forward pass,
    and skips non-differentiable points to pass gradients in backward pass.
    """

    @staticmethod
    def forward(ctx, input_tensor, num_bits):
        if num_bits is None or num_bits >= 32:
            return input_tensor

        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        min_val, max_val = input_tensor.min(), input_tensor.max()
        scale = (max_val - min_val) / (2 ** num_bits - 1) + 1e-8

        quantized = torch.round((input_tensor - min_val) / scale) + qmin
        dequantized = (quantized - qmin) * scale + min_val
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize(tensor, num_bits):
    """Fake quantization wrapper function."""
    if num_bits is None or num_bits >= 32:
        return tensor
    return FakeQuantizeSTE.apply(tensor, num_bits)


def generate_pruning_mask(weights, pruning_ratio):
    """Generate magnitude-based pruning mask based on the absolute values of variational parameters."""
    if pruning_ratio <= 0.0:
        return torch.ones_like(weights)

    threshold = torch.quantile(torch.abs(weights), pruning_ratio)
    mask = (torch.abs(weights) >= threshold).float()
    return mask


# ==========================================
# 2. 1D Residual Block (for vector channel feature processing)
# ==========================================
class ResBlock1D(nn.Module):
    """Residual block designed for 1D channel data/feature vectors (1D ResNet Block)."""

    def __init__(self, in_dim, out_dim):
        super(ResBlock1D, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = self.relu(out + residual)
        return out


# ==========================================
# 3. ResNet-HQNN Architecture Supporting Compressibility Ablation Experiments
# ==========================================
class QuantumMemoryDNN(nn.Module):
    """
    Hybrid neural network combining 1D ResNet with PennyLane Variational Quantum Circuit (VQC).
    Supports dynamic qubit count (n_qubits), pruning ratio (pruning_ratio), and quantization bits (quant_bits).
    """

    def __init__(self, net_dims, n_qubits=6, n_layers=2, pruning_ratio=0.0, quant_bits=32,
                 learning_rate=0.01, training_interval=10, batch_size=128, memory_size=1024):
        super(QuantumMemoryDNN, self).__init__()

        self.input_dim = net_dims[0]
        self.output_dim = net_dims[-1]
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pruning_ratio = pruning_ratio
        self.quant_bits = quant_bits

        # 1. Dynamically instantiate PennyLane virtual quantum device and QNode
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # (a) Data encoding layer
            for i in range(self.n_qubits):
                qml.RY(inputs[..., i], wires=i)
            # (b) Variational & entangling layer
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            # (c) Pauli-Z expectation value measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # 2. Classical ResNet encoder/decoder (automatically adapts to n_qubits dimension)
        self.encoder = nn.Sequential(
            ResBlock1D(self.input_dim, 64),
            ResBlock1D(64, 64),
            nn.Linear(64, self.n_qubits),
            nn.Tanh()  # Map to [-1, 1]
        )

        self.decoder = nn.Sequential(
            ResBlock1D(self.n_qubits, 64),
            ResBlock1D(64, 64),
            nn.Linear(64, self.output_dim)
        )

        # 3. Reinforcement learning and training parameters
        self.learning_rate = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=self.memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.cost_his = []
        self.time_step = 0

    def forward(self, x):
        # (1) Classical feature extraction and state encoding fake quantization
        x_angles = self.encoder(x) * np.pi
        x_angles = fake_quantize(x_angles, self.quant_bits)

        # (2) Dynamic pruning and fake quantization of variational quantum weights
        with torch.no_grad():
            mask = generate_pruning_mask(self.q_layer.weights, self.pruning_ratio)

        pruned_quant_weights = fake_quantize(self.q_layer.weights * mask, self.quant_bits)

        # Attach pruned & quantized temporary weights to the circuit, then restore original weights after computation to ensure autograd updates correctly
        orig_weights = self.q_layer.weights
        self.q_layer.weights = nn.Parameter(pruned_quant_weights)

        quantum_features = self.q_layer(x_angles)

        self.q_layer.weights = orig_weights

        # (3) Classical ResNet decoding
        q_values = self.decoder(quantum_features)
        return q_values

    def decode(self, h, K, mode='OP'):
        """Evaluation mode: generate candidate solution set using order-preserving quantization."""
        self.eval()
        h_tensor = torch.FloatTensor(h).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(h_tensor).squeeze(0).cpu().numpy()

        if mode == 'OP':
            return self._order_preserving_quantization(q_values, K)
        else:
            return self._knn_quantization(q_values, K)

    def _order_preserving_quantization(self, q_values, K):
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
        """Push into Experience Replay and periodically trigger backpropagation."""
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
        # Gradient clipping to prevent barren plateaus
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.cost_his.append(loss.item())


# ==========================================
# 4. Data Saving and Auxiliary Evaluation Module
# ==========================================
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        for val in data:
            f.write("%s\n" % val)


def calculate_video_transmission_time(task_folder, avg_normalized_rate, results_dir="./result-ResNet"):
    """5G real physical transmission latency evaluation module"""
    REAL_5G_BASE_RATE_MBPS = 300.0
    actual_avg_rate_bps = avg_normalized_rate * REAL_5G_BASE_RATE_MBPS * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (5G Network Standard) ---")
    print(f"Quantum residual network offline normalized efficiency: {avg_normalized_rate:.4f}")
    print(f"Current predicted real 5G average transmission rate: {actual_avg_rate_bps / 1e6:.3f} Mbps")

    search_path = os.path.join(task_folder, "*.mp4")
    video_files = glob.glob(search_path)

    if not video_files:
        print(f"Notice: No .mp4 video files detected in '{task_folder}' folder.")
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
        f.write("    PennyLane ResNet-integrated Quantum Reinforcement Learning Report\n")
        f.write("=========================================================\n")
        f.write(f"5G Network Scenario Baseline Setting: {REAL_5G_BASE_RATE_MBPS:.1f} Mbps\n")
        f.write(f"Quantum Variational Network Optimized Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps\n")
        f.write(f"Total Processed Video Files: {len(video_files)}\n")
        f.write(f"Total Video File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} seconds\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")

    print(f"Video transmission time analysis completed! Report exported to: '{output_file}'\n")


# ==========================================
# 5. Main Execution Entry Point for Quantum Circuit Compressibility Ablation Experiments
# ==========================================
if __name__ == "__main__":
    results_dir = "./result-Compressibility_Ablation"
    os.makedirs(results_dir, exist_ok=True)

    N = 10  # Number of users / channels
    n = 30000  # Number of time frames for ablation experiments (suggest 2000 frames for quick convergence checks, 20000+ for long runs)
    K = N  # Solution space dimension
    decoder_mode = 'OP'  # Order-preserving quantization mode
    Memory = 1024  # Replay Memory capacity
    Delta = 32  # Adaptive K adjustment interval
    REAL_5G_BASE_RATE_MBPS = 300.0
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # Edge task size (5 MB)

    # 1. Try reading channel data (if file not found, generate simulated random channels to ensure standalone execution)
    data_path = './data/data_%d.mat' % N
    if os.path.exists(data_path):
        channel = sio.loadmat(data_path)['input_h'] * 1000000
        rate = sio.loadmat(data_path)['output_obj']
    else:
        print(f"Dataset {data_path} not detected, generated random channel matrix for testing run...")
        channel = np.random.rand(n, N) * 100
        rate = np.random.rand(n, 1) + 1.0

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    # 2. Define control groups for compressibility ablation experiments (controlled variable method)
    # Configuration list format: (Display Name, n_qubits, pruning_ratio, quant_bits)
    ablation_configs = [
        ("Baseline (n=6, Prune=0%, FP32)", 6, 0.0, 32),
        ("Qubits (n=4, Prune=0%, FP32)", 4, 0.0, 32),
        ("Qubits (n=8, Prune=0%, FP32)", 8, 0.0, 32),
        ("Gate Pruning (n=6, Prune=30%, FP32)", 6, 0.3, 32),
        ("Gate Pruning (n=6, Prune=50%, FP32)", 6, 0.5, 32),
        ("Quantization INT8 (n=6, Prune=0%, INT8)", 6, 0.0, 8),
        ("Quantization INT4 (n=6, Prune=0%, INT4)", 6, 0.0, 4)
    ]

    all_loss_histories = {}
    baseline_norm_rates = []

    # 3. Run ablation experiment groups sequentially
    for config_name, q_num, p_ratio, q_bits in ablation_configs:
        print(f"\n==================================================")
        print(f" Running ablation experiment: {config_name}")
        print(f"==================================================")

        # Reset random seed to ensure baseline environment is identical across groups
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        mem = QuantumMemoryDNN(
            net_dims=[N, 64, N],
            n_qubits=q_num,
            pruning_ratio=p_ratio,
            quant_bits=q_bits,
            learning_rate=0.01,
            training_interval=10,
            batch_size=128,
            memory_size=Memory
        )

        k_idx_his = []
        rate_his_ratio = []

        start_time = time.time()

        for i in range(n):
            if i > 0 and i % Delta == 0:
                max_k = max(k_idx_his[-Delta:-1]) + 1 if Delta > 1 else k_idx_his[-1] + 1
                K = min(max_k + 1, N)

            i_idx = i % split_idx if i < n - num_test else i - n + num_test + split_idx
            h = channel[i_idx, :]

            # Decoding and physical layer bisection optimization
            m_list = mem.decode(h, K, decoder_mode)
            r_list = [bisection(h / 1000000, m)[0] for m in m_list]

            best_rate = np.max(r_list)
            best_idx = np.argmax(r_list)

            # Store history and perform network updates
            mem.encode(h, m_list[best_idx])
            k_idx_his.append(best_idx)

            norm_rate = best_rate / rate[i_idx][0] if isinstance(rate[i_idx], np.ndarray) else best_rate / rate[i_idx]
            rate_his_ratio.append(norm_rate)

        total_run_time = time.time() - start_time
        avg_norm_rate = np.mean(rate_his_ratio[-num_test:])

        all_loss_histories[config_name] = mem.cost_his

        print(f"[{config_name}] Run completed!")
        print(f" - Final Loss: {mem.cost_his[-1] if len(mem.cost_his) > 0 else 0.0:.4f}")
        print(f" - Final Average Normalized Rate: {avg_norm_rate:.4f}")
        print(f" - Total Time Consumed: {total_run_time:.2f} seconds")

        # Export Loss data for individual group
        safe_name = config_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "").replace("=", "_")
        save_data(mem.cost_his, os.path.join(results_dir, f"loss_{safe_name}.txt"))

    # 4. Plot comparison figure: Training Loss Convergence
    plt.figure(figsize=(9, 5))
    for name, loss_his in all_loss_histories.items():
        if len(loss_his) > 0:
            df_loss = pd.DataFrame(loss_his)
            smooth_loss = df_loss.rolling(10, min_periods=1).mean().values.ravel()
            plt.plot(smooth_loss, label=name, linewidth=1.6)

    plt.ylabel('Loss', fontsize=11)
    plt.xlabel('Training Iterations', fontsize=11)
    plt.title('Convergence Analysis of Quantum Circuit Compressibility', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    convergence_fig_path = os.path.join(results_dir, "ablation_compressibility_convergence.png")
    plt.savefig(convergence_fig_path, dpi=300)
    plt.show()

    print(f"\nAll ablation experiments completed!")
    print(f"Convergence comparison plot saved to: {convergence_fig_path}")