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
import copy
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import psutil
from thop import profile  # Used for calculating FLOPs

# ==========================================
# Core Import: PennyLane Variational Quantum Circuit (VQC) Library
# ==========================================
import pennylane as qml

# Try importing physical layer optimization algorithm; provide fallback implementation if missing locally
try:
    from optimization import bisection
except ImportError:
    def bisection(h_scaled, m):
        """Physical layer bisection operator (degraded/fallback implementation)"""
        rate = np.sum(h_scaled * m) * np.log2(1 + np.sum(h_scaled * m))
        return [max(0.1, rate)]

# Set random seed to ensure reproducibility of quantum and RL experiments
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1. Variational Quantum Circuit (VQC) Architecture Standardized to Paper
# ==========================================
n_qubits = 6  # Keep number of qubits consistent with Light-QDRL
n_layers = 2  # Keep circuit depth consistent with Light-QDRL

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Variational quantum circuit architecture shared by traditional QDQN
    """
    # 1. Encoding layer
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)

    # 2. Variational and entanglement layer
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # 3. Expectation measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==========================================
# 2. Traditional QDQN Network Architecture (Q-Network)
# ==========================================
class QuantumQNetwork(nn.Module):
    """
    Traditional QDQN core network: Input channel state s, output Q-values for all 2^N combination actions
    """

    def __init__(self, input_dim, output_action_dim, n_qubits):
        super(QuantumQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_action_dim = output_action_dim  # 2^N action space size
        self.n_qubits = n_qubits

        # Classical encoding layer
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.n_qubits),
            nn.Tanh()
        )

        # Variational quantum layer
        weight_shapes = {"weights": (n_layers, self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classical decoding layer: maps to discrete action space Q-values (2^N dimension)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_action_dim)
        )

    def forward(self, x):
        x_angles = self.encoder(x) * np.pi
        q_features = self.q_layer(x_angles)
        q_values = self.decoder(q_features)
        return q_values


# ==========================================
# 3. Traditional QDQN Agent
# ==========================================
class TraditionalQDQNAgent:
    """
    Traditional QDQN trainer (with Target Network and Epsilon-Greedy exploration)
    """

    def __init__(self, n_users, learning_rate=0.01, gamma=0.9, memory_size=1024, batch_size=128,
                 target_update_freq=50):
        self.n_users = n_users
        self.action_dim = 2 ** n_users  # Total discrete actions 2^N
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq

        # Epsilon-Greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        # Main network and target network (Dual-net)
        self.eval_net = QuantumQNetwork(n_users, self.action_dim, n_qubits)
        self.target_net = QuantumQNetwork(n_users, self.action_dim, n_qubits)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=self.memory_size)
        self.cost_his = []
        self.learn_step_counter = 0

    def idx_to_action(self, action_idx):
        """Convert discrete action index back to N-dimensional binary action vector"""
        binary_str = format(action_idx, f'0{self.n_users}b')
        return np.array([int(b) for b in binary_str])

    def select_action(self, state):
        """Epsilon-Greedy policy action selection"""
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            self.eval_net.eval()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.eval_net(state_tensor)
                action_idx = torch.argmax(q_values, dim=1).item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action_idx, self.idx_to_action(action_idx)

    def store_transition(self, s, a, r, s_next):
        """Store TD transition tuple into Replay Buffer"""
        self.memory.append((s, a, r, s_next))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.eval_net.train()
        batch = random.sample(self.memory, self.batch_size)
        s_b, a_b, r_b, s_next_b = zip(*batch)

        s_tensor = torch.FloatTensor(np.array(s_b))
        a_tensor = torch.LongTensor(a_b).unsqueeze(1)
        r_tensor = torch.FloatTensor(r_b).unsqueeze(1)
        s_next_tensor = torch.FloatTensor(np.array(s_next_b))

        q_eval = self.eval_net(s_tensor).gather(1, a_tensor)

        with torch.no_grad():
            q_next = self.target_net(s_next_tensor).max(1)[0].unsqueeze(1)
            q_target = r_tensor + self.gamma * q_next

        loss = F.mse_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 0.5)
        self.optimizer.step()

        self.cost_his.append(loss.item())
        self.learn_step_counter += 1

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_his, color='crimson', linewidth=1.5)
        plt.ylabel('Traditional QDQN Loss')
        plt.xlabel('Training Iterations')
        plt.title('Convergence Curve of Traditional QDQN')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()


# ==========================================
# 4. Edge Energy Efficiency and Computational Efficiency Evaluation Module (Edge Benchmark Tools)
# ==========================================
class EdgeEfficiencyEvaluator:
    """Edge energy efficiency, inference latency, and FLOPs evaluator (for traditional QDQN version)"""

    def __init__(self, agent, device="cpu"):
        self.agent = agent
        self.model = agent.eval_net
        self.device = device
        self.model.to(self.device)

    def measure_inference_latency(self, dummy_input, num_warmup=100, num_runs=1000):
        """1. Measure pure neural network inference latency (Inference Time, unit: ms)"""
        self.model.eval()
        dummy_input = dummy_input.to(self.device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(dummy_input)

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                _ = self.model(dummy_input)

                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        print(f"[QDQN Latency] Avg Inference Time: {avg_latency:.4f} ms | P95 Latency: {p95_latency:.4f} ms")
        return avg_latency, p95_latency

    def measure_memory_and_flops(self, dummy_input):
        """2. Measure FLOPs and Memory Footprint

        Note: Since QDQN Dual-net architecture maintains both Eval-Net and Target-Net, the actual static memory is the sum of both networks.
        """
        self.model.eval()
        dummy_input = dummy_input.to(self.device)

        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

        # Static model weight memory size
        param_mem_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_mem_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        # QDQN contains Eval-Net and Target-Net mirrored networks, so parameter count and memory need to be multiplied by 2
        total_model_mem_mb = 2 * (param_mem_bytes + buffer_mem_bytes) / (1024 ** 2)

        process = psutil.Process(os.getpid())
        ram_usage_mb = process.memory_info().rss / (1024 ** 2)

        print(f"[QDQN Footprint] MACs/FLOPs: {flops / 1e6:.4f} MFLOPs | Eval Params: {params} (Dual Net total: {params*2})")
        print(f"[QDQN Footprint] Dual-Net Static Size: {total_model_mem_mb:.4f} MB | RAM Usage: {ram_usage_mb:.2f} MB")
        return flops, total_model_mem_mb, ram_usage_mb

    def measure_finetuning_energy(self, dummy_batch, num_steps=50, power_watt_estimate=5.0):
        """3. QDQN edge online fine-tuning energy consumption evaluation (Fine-tuning Energy)"""
        self.model.train()
        s_tensor, target_q = dummy_batch
        s_tensor, target_q = s_tensor.to(self.device), target_q.to(self.device)

        step_times = []
        for _ in range(num_steps):
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            self.agent.optimizer.zero_grad()
            pred_q = self.model(s_tensor)
            loss = F.mse_loss(pred_q, target_q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.agent.optimizer.step()

            if self.device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            step_times.append(end - start)

        avg_step_time = np.mean(step_times)
        energy_per_step_j = power_watt_estimate * avg_step_time
        energy_per_step_mwh = (energy_per_step_j / 3600.0) * 1000.0

        print(f"[QDQN Energy Benchmark] Avg Step Time: {avg_step_time * 1000:.2f} ms | Energy/Step: {energy_per_step_j:.4f} J ({energy_per_step_mwh:.6f} mWh)")
        return step_times, energy_per_step_j


# ==========================================
# 5. Data Saving and Plotting Utility Functions
# ==========================================
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        for val in data:
            f.write("%s\n" % val)


def plot_normalized_rate(rate_his_ratio, save_path=None, rolling_intv=50):
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
    plt.title('Normalized Computation Rate vs. Time Frames (Traditional QDQN)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_throughput(throughput_his, save_path=None, rolling_intv=50):
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
    plt.title('System Throughput over Time Frames (Traditional QDQN)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cdf_latency(latency_his, save_path=None):
    sorted_latency = np.sort(np.asarray(latency_his))
    cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(sorted_latency, cdf, color='#d62728', linewidth=2.0, label='Traditional QDQN Latency CDF')

    plt.ylabel('Cumulative Probability (CDF)')
    plt.xlabel('Transmission & Processing Latency (ms)')
    plt.title('Cumulative Distribution Function (CDF) of Latency (Traditional QDQN)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_video_transmission_time(task_folder, avg_normalized_rate, results_dir="./result-QDQN"):
    REAL_5G_BASE_RATE_MBPS = 300.0
    actual_avg_rate_bps = avg_normalized_rate * REAL_5G_BASE_RATE_MBPS * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (Traditional QDQN) ---")
    print(f"Traditional QDQN Normalized Efficiency: {avg_normalized_rate:.4f}")
    print(f"Predicted Real 5G Average Transmission Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps")

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
        f.write("    Traditional QDQN Evaluation Report\n")
        f.write("=========================================================\n")
        f.write(f"5G Network Baseline Setting: {REAL_5G_BASE_RATE_MBPS:.1f} Mbps\n")
        f.write(f"Optimized Transmission Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps\n")
        f.write(f"Processed Video Count: {len(video_files)}\n")
        f.write(f"Total File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} seconds\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")
        f.write("=========================================================\n")

    print(f"Transmission time analysis exported to: '{output_file}'\n")


# ==========================================
# 6. Main Program Execution Entry Point
# ==========================================
if __name__ == "__main__":
    results_dir = "./result-QDQN"
    os.makedirs(results_dir, exist_ok=True)

    N = 5  # Number of users (QDQN benchmark set to N=5 due to 2^N dimensionality explosion)
    n = 2000  # Total time frames (set to 2000 frames for convenient testing)
    Memory = 1024  # Replay buffer size
    REAL_5G_BASE_RATE_MBPS = 300.0  # 5G baseline bandwidth
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # 5MB task size

    print('#user = %d, #channel = %d, Memory = %d (Traditional QDQN Benchmarking)' % (N, n, Memory))

    # 1. Load channel data (automatically simulate if file is missing)
    data_path = './data/data_%d.mat' % N
    if os.path.exists(data_path):
        channel = sio.loadmat(data_path)['input_h']
        rate = sio.loadmat(data_path)['output_obj']
    else:
        print(f"Data file {data_path} not found. Automatically generated simulated channels for Benchmark testing.")
        channel = np.random.exponential(scale=1.0, size=(n, N))
        rate = np.ones((n, 1)) * 50.0

    channel = channel * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    # 2. Instantiate Traditional QDQN Agent
    agent = TraditionalQDQNAgent(n_users=N, learning_rate=0.01, memory_size=Memory, batch_size=128)

    # ==========================================
    # Execute Edge Energy Efficiency and Computational Efficiency Benchmark (QDQN Version)
    # ==========================================
    print("\n" + "=" * 50)
    print("      Executing Traditional QDQN Edge Energy Efficiency Benchmark...")
    print("=" * 50)

    evaluator = EdgeEfficiencyEvaluator(agent, device="cpu")

    # (1) Simulate single edge input (Batch_size = 1, Channel dimension N)
    dummy_input = torch.randn(1, N)

    # Measure inference latency (Inference Latency)
    avg_latency, p95_latency = evaluator.measure_inference_latency(dummy_input)

    # Measure FLOPs and Dual-Net Memory Footprint
    flops, model_mem_mb, ram_mb = evaluator.measure_memory_and_flops(dummy_input)

    # (2) Simulate QDQN Fine-tuning Batch (Batch_size = 128, Action output dimension 2^N)
    dummy_batch = (torch.randn(128, N), torch.randn(128, agent.action_dim))
    step_times, energy_per_step = evaluator.measure_finetuning_energy(dummy_batch, power_watt_estimate=5.0)

    # (3) Export Efficiency Benchmark text report
    with open(os.path.join(results_dir, "edge_efficiency_metrics.txt"), "w") as f:
        f.write("====================================================\n")
        f.write("      Traditional QDQN Edge Efficiency Benchmark\n")
        f.write("====================================================\n")
        f.write(f"Inference Latency (Avg) : {avg_latency:.4f} ms\n")
        f.write(f"Inference Latency (P95) : {p95_latency:.4f} ms\n")
        f.write(f"FLOPs per Inference     : {flops / 1e6:.4f} MFLOPs\n")
        f.write(f"Static Model Memory Size: {model_mem_mb:.4f} MB (Dual Net: Eval + Target)\n")
        f.write(f"Runtime RAM Occupancy   : {ram_mb:.2f} MB\n")
        f.write(f"Fine-Tuning Time/Step   : {np.mean(step_times) * 1000:.2f} ms\n")
        f.write(f"Estimated Energy/Step   : {energy_per_step:.4f} Joules\n")
        f.write("====================================================\n")

    # (4) Plot and save QDQN fine-tuning energy curve
    plt.figure(figsize=(7, 4))
    energy_steps_mj = (np.array(step_times) * 5.0) * 1000.0  # mJ
    plt.plot(energy_steps_mj, color='#d32f2f', linewidth=1.5, marker='s', markersize=3)
    plt.ylabel('Energy Consumption per Step (mJ)')
    plt.xlabel('Fine-Tuning Steps (Traditional QDQN)')
    plt.title('QDQN Fine-Tuning Energy Profile @ 5W TDP')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fine_tuning_energy_profile.png"), dpi=300)
    plt.close()

    print("=" * 50 + "\n")

    # ==========================================
    # 3. Run online RL optimization loop
    # ==========================================
    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    throughput_his = []
    latency_his = []

    for i in range(n):
        if i % (n // 10) == 0:
            print("%0.1f" % (i / n))

        if i < n - num_test:
            i_idx = i % split_idx
            next_i_idx = (i + 1) % split_idx
        else:
            i_idx = i - n + num_test + split_idx
            next_i_idx = min(i_idx + 1, len(channel) - 1)

        h = channel[i_idx, :]
        h_next = channel[next_i_idx, :]

        # (a) Action selection (Epsilon-Greedy over 2^N Discrete Actions)
        action_idx, binary_action = agent.select_action(h)

        # (b) Calculate physical layer Rate for current action (Immediate Reward)
        reward = bisection(h / 1000000, binary_action)[0]

        # (c) Store into Replay Buffer and update Q-Network
        agent.store_transition(h, action_idx, reward, h_next)
        agent.learn()

        # (d) Record performance metrics
        norm_rate = reward / rate[i_idx][0]
        current_throughput_mbps = norm_rate * REAL_5G_BASE_RATE_MBPS
        current_latency_ms = (TASK_SIZE_BITS / (current_throughput_mbps * 1e6)) * 1000.0

        rate_his.append(reward)
        rate_his_ratio.append(norm_rate)
        throughput_his.append(current_throughput_mbps)
        latency_his.append(current_latency_ms)

    total_time = time.time() - start_time

    avg_normalized_rate = sum(rate_his_ratio[-num_test: -1]) / num_test
    print("Averaged normalized computation rate:", avg_normalized_rate)
    print('Total time consumed: %s' % total_time)
    print('Average time per channel: %s' % (total_time / n))

    # 4. Save data to ./result-QDQN/
    save_data(rate_his_ratio, os.path.join(results_dir, "normalized_computation_rate.txt"))
    save_data(throughput_his, os.path.join(results_dir, "throughput_mbps.txt"))
    save_data(latency_his, os.path.join(results_dir, "latency_ms.txt"))
    save_data(agent.cost_his, os.path.join(results_dir, "cost_his.txt"))

    np.save(os.path.join(results_dir, "normalized_computation_rate.npy"), np.array(rate_his_ratio))
    np.save(os.path.join(results_dir, "throughput_mbps.npy"), np.array(throughput_his))
    np.save(os.path.join(results_dir, "latency_ms.npy"), np.array(latency_his))

    # 5. Plot and save curves
    agent.plot_cost(save_path=os.path.join(results_dir, "qdqn_loss_convergence.png"))
    plot_normalized_rate(rate_his_ratio, save_path=os.path.join(results_dir, "normalized_computation_rate.png"))
    plot_throughput(throughput_his, save_path=os.path.join(results_dir, "throughput.png"))
    plot_cdf_latency(latency_his, save_path=os.path.join(results_dir, "cdf_of_latency.png"))

    # 6. Real video transmission time evaluation
    calculate_video_transmission_time(task_folder="./task",
                                      avg_normalized_rate=avg_normalized_rate,
                                      results_dir=results_dir)