import scipy.io as sio  # import scipy.io for .mat file I/O
import numpy as np  # import numpy
import torch
import random
import time
import os  # Used for folder operations and obtaining file sizes
import sys
import glob  # Used for matching video filenames
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import psutil  # Used for memory/system resource evaluation

# Retain physical layer bisection optimization algorithm
try:
    from optimization import bisection
except ImportError:
    def bisection(h_scaled, m):
        """Physical layer bisection operator (fallback/degraded implementation)"""
        rate = np.sum(h_scaled * m) * np.log2(1 + np.sum(h_scaled * m))
        return [max(0.1, rate)]

# Set random seed for experiment reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

plt.rcParams.update({
    'font.size': 14,          # Global base font
    'axes.labelsize': 16,     # Axis labels xlabel/ylabel
    'axes.titlesize': 18,     # Chart title
    'xtick.labelsize': 14,    # X-axis tick labels
    'ytick.labelsize': 14,    # Y-axis tick labels
    'legend.fontsize': 14     # Legend
})


# ==========================================
# 1. Channel State Discretization Module (State Discretizer)
# ==========================================
class StateDiscretizer:
    """
    Maps continuous channel gain vector h (dimension N) to finite discrete state (Tuple)
    Used to establish Q-table indices for tabular Q-Learning
    """

    def __init__(self, n_users, n_bins=5):
        self.n_users = n_users
        self.n_bins = n_bins
        self.bins = []

    def fit(self, channel_data):
        """Dynamically fits bin boundaries according to global channel distribution"""
        self.bins = []
        for dim in range(self.n_users):
            dim_data = channel_data[:, dim]
            b = np.linspace(dim_data.min(), dim_data.max(), self.n_bins + 1)[1:-1]
            self.bins.append(b)

    def discretize(self, h):
        """Maps a single channel state h to a discrete state tuple"""
        discrete_s = []
        for dim in range(self.n_users):
            idx = np.digitize(h[dim], self.bins[dim])
            discrete_s.append(idx)
        return tuple(discrete_s)


# ==========================================
# 2. Traditional Q-Learning Agent (Tabular Q-Learning)
# ==========================================
class TraditionalQLearningAgent:
    """
    Traditional Q-Learning Agent (Based on Q-Table and Epsilon-Greedy Strategy)
    """

    def __init__(self, n_users, learning_rate=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        self.n_users = n_users
        self.action_dim = 2 ** n_users  # Discrete action space 2^N
        self.lr = learning_rate
        self.gamma = gamma

        # Epsilon-Greedy exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: Uses dictionary to store mapping from discrete state to Q-values (sparse table)
        self.q_table = {}
        self.cost_his = []  # Record TD Error squared as Convergence Loss

    def get_q_values(self, state_tuple):
        """Look up or initialize the Q-value array for the current state"""
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_dim)
        return self.q_table[state_tuple]

    def idx_to_action(self, action_idx):
        """Convert discrete action index back to N-dimensional binary action vector"""
        binary_str = format(action_idx, f'0{self.n_users}b')
        return np.array([int(b) for b in binary_str])

    def select_action(self, state_tuple):
        """Epsilon-Greedy action selection"""
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            q_vals = self.get_q_values(state_tuple)
            action_idx = np.argmax(q_vals)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action_idx, self.idx_to_action(action_idx)

    def learn(self, s_tuple, action_idx, reward, s_next_tuple):
        """Standard Bellman tabular update equation (Bellman Equation Update)"""
        q_vals = self.get_q_values(s_tuple)
        q_vals_next = self.get_q_values(s_next_tuple)

        predict = q_vals[action_idx]
        target = reward + self.gamma * np.max(q_vals_next)
        td_error = target - predict

        # Update entry in Q-table
        self.q_table[s_tuple][action_idx] += self.lr * td_error

        # Record TD Error squared as convergence loss
        self.cost_his.append(td_error ** 2)

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_his, color='darkorange', linewidth=1.2, alpha=0.7)
        plt.ylabel('TD Error Squared')
        plt.xlabel('Learning Steps')
        plt.title('Convergence Curve of Traditional Q-Learning')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()


# ==========================================
# Edge Energy Efficiency and Computational Efficiency Benchmark Module (Edge Benchmark Tools for Q-Learning)
# ==========================================
class EdgeEfficiencyEvaluator:
    """Edge energy efficiency, inference latency, and memory overhead evaluator (Traditional Q-Learning Tabular Standard Version)"""

    def __init__(self, agent, discretizer):
        self.agent = agent
        self.discretizer = discretizer

    def measure_inference_latency(self, dummy_h, num_warmup=100, num_runs=1000):
        """1. Measure single-step decision inference latency (includes state discretization + Q-table lookup/Argmax, unit: ms)"""
        for _ in range(num_warmup):
            s_tuple = self.discretizer.discretize(dummy_h)
            _ = self.agent.select_action(s_tuple)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()

            s_tuple = self.discretizer.discretize(dummy_h)
            _ = self.agent.select_action(s_tuple)

            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        print(f"[Q-Learning Latency] Avg Inference Time: {avg_latency:.4f} ms | P95 Latency: {p95_latency:.4f} ms")
        return avg_latency, p95_latency

    def measure_memory_and_flops(self):
        """2. Measure Q-table structural overhead and runtime RAM footprint (Memory Footprint)"""
        # Calculate Q-Table dictionary static memory overhead (Bytes -> MB)
        q_table_bytes = sys.getsizeof(self.agent.q_table)
        for k, v in self.agent.q_table.items():
            q_table_bytes += sys.getsizeof(k) + v.nbytes

        q_table_mem_mb = q_table_bytes / (1024 ** 2)

        # FLOPs/MACs for tabular methods degrade to N bin lookups + 2^N array lookups (expressed in MFLOPs)
        estimated_flops = (self.discretizer.n_users * self.discretizer.n_bins + self.agent.action_dim)

        process = psutil.Process(os.getpid())
        ram_usage_mb = process.memory_info().rss / (1024 ** 2)

        print(f"[Q-Learning Footprint] Effective Ops: {estimated_flops / 1e6:.6f} MFLOPs | Q-Table Size: {len(self.agent.q_table)} entries")
        print(f"[Q-Learning Footprint] Static Table Size: {q_table_mem_mb:.4f} MB | Runtime RAM Usage: {ram_usage_mb:.2f} MB")
        return estimated_flops, q_table_mem_mb, ram_usage_mb

    def measure_finetuning_energy(self, dummy_step_data, num_steps=50, power_watt_estimate=5.0):
        """3. Q-Learning edge online table update energy consumption evaluation (TD-Update Energy)"""
        s_tuple, action_idx, reward, s_next_tuple = dummy_step_data

        step_times = []
        for _ in range(num_steps):
            start = time.perf_counter()

            self.agent.learn(s_tuple, action_idx, reward, s_next_tuple)

            end = time.perf_counter()
            step_times.append(end - start)

        avg_step_time = np.mean(step_times)
        energy_per_step_j = power_watt_estimate * avg_step_time
        energy_per_step_mwh = (energy_per_step_j / 3600.0) * 1000.0

        print(f"[Q-Learning Energy Benchmark] Avg Step Time: {avg_step_time * 1000:.4f} ms | Energy/Step: {energy_per_step_j:.6f} J ({energy_per_step_mwh:.8f} mWh)")
        return step_times, energy_per_step_j


# ==========================================
# 4. Data Saving and Plotting Utility Functions
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
    plt.title('Normalized Computation Rate vs. Time Frames (Traditional Q-Learning)')
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
    plt.title('System Throughput over Time Frames (Traditional Q-Learning)')
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
    plt.plot(sorted_latency, cdf, color='#d62728', linewidth=2.0, label='Traditional Q-Learning Latency CDF')

    plt.ylabel('Cumulative Probability (CDF)')
    plt.xlabel('Transmission & Processing Latency (ms)')
    plt.title('Cumulative Distribution Function (CDF) of Latency (Traditional Q-Learning)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_video_transmission_time(task_folder, avg_normalized_rate, results_dir="./result-QLearning"):
    REAL_5G_BASE_RATE_MBPS = 300.0
    actual_avg_rate_bps = avg_normalized_rate * REAL_5G_BASE_RATE_MBPS * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (Traditional Q-Learning) ---")
    print(f"Traditional Q-Learning Normalized Efficiency: {avg_normalized_rate:.4f}")
    print(f"Predicted Real 5G Average Transmission Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps")

    search_path = os.path.join(task_folder, "*.mp4")
    video_files = glob.glob(search_path)

    if not video_files:
        print(f"Notice: No .mp4 video files detected in '{task_folder}' directory.")
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
        f.write(" Traditional Q-Learning Evaluation Report\n")
        f.write("=========================================================\n")
        f.write(f"5G Network Scenario Baseline Setting: {REAL_5G_BASE_RATE_MBPS:.1f} Mbps\n")
        f.write(f"Optimized Transmission Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps\n")
        f.write(f"Total Video Files Processed: {len(video_files)}\n")
        f.write(f"Total Video File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} s\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")
        f.write("=========================================================\n")

    print(f"Video transmission time analysis exported to: '{output_file}'\n")


# ==========================================
# 5. Main Program Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Create independent results storage directory ./result-QLearning
    results_dir = "./result-QLearning"
    os.makedirs(results_dir, exist_ok=True)

    N = 30  # Number of users
    n = 30000  # Total number of time frames
    REAL_5G_BASE_RATE_MBPS = 300.0  # Baseline 5G bandwidth rate (Mbps)
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # Edge task size (5MB)

    print('#user = %d, #channel = %d (Traditional Q-Learning Benchmarking)' % (N, n))

    # 1. Load channel data and baseline rate
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

    # 2. State discretizer fitting and agent initialization
    discretizer = StateDiscretizer(n_users=N, n_bins=5)
    discretizer.fit(channel)

    agent = TraditionalQLearningAgent(n_users=N, learning_rate=0.1, gamma=0.9)

    # ==========================================
    # Execute Edge Energy Efficiency and Computational Efficiency Benchmark (Traditional Q-Learning Baseline)
    # ==========================================
    print("\n" + "=" * 50)
    print("  Executing Traditional Q-Learning Edge Energy Efficiency Benchmark...")
    print("=" * 50)

    evaluator = EdgeEfficiencyEvaluator(agent, discretizer)

    # (1) Simulate single edge input channel (channel dimension N)
    dummy_h = channel[0, :]

    # Measure inference latency (Inference Latency)
    avg_latency, p95_latency = evaluator.measure_inference_latency(dummy_h)

    # Measure Q-table memory overhead and RAM usage
    flops, model_mem_mb, ram_mb = evaluator.measure_memory_and_flops()

    # (2) Simulate single-step TD-Update tabular update
    dummy_s_tuple = discretizer.discretize(dummy_h)
    dummy_s_next_tuple = discretizer.discretize(channel[1, :])
    dummy_step_data = (dummy_s_tuple, 0, 1.0, dummy_s_next_tuple)
    step_times, energy_per_step = evaluator.measure_finetuning_energy(dummy_step_data, power_watt_estimate=5.0)

    # (3) Export Efficiency Benchmark text report
    with open(os.path.join(results_dir, "edge_efficiency_metrics.txt"), "w") as f:
        f.write("====================================================\n")
        f.write("    Traditional Q-Learning Edge Efficiency Benchmark\n")
        f.write("====================================================\n")
        f.write(f"Inference Latency (Avg) : {avg_latency:.4f} ms\n")
        f.write(f"Inference Latency (P95) : {p95_latency:.4f} ms\n")
        f.write(f"FLOPs per Inference     : {flops / 1e6:.6f} MFLOPs (Tabular Ops)\n")
        f.write(f"Static Q-Table Size     : {model_mem_mb:.4f} MB\n")
        f.write(f"Runtime RAM Occupancy   : {ram_mb:.2f} MB\n")
        f.write(f"Fine-Tuning Time/Step   : {np.mean(step_times) * 1000:.4f} ms\n")
        f.write(f"Estimated Energy/Step   : {energy_per_step:.6f} Joules\n")
        f.write("====================================================\n")

    # (4) Plot and save Traditional Q-Learning fine-tuning power consumption curve
    plt.figure(figsize=(7, 4))
    energy_steps_mj = (np.array(step_times) * 5.0) * 1000.0  # mJ
    plt.plot(energy_steps_mj, color='#f57c00', linewidth=1.5, marker='^', markersize=3)
    plt.ylabel('Energy Consumption per Step (mJ)')
    plt.xlabel('Fine-Tuning Steps (Q-Learning Table Update)')
    plt.title('Traditional Q-Learning Fine-Tuning Energy Profile @ 5W TDP')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fine_tuning_energy_profile.png"), dpi=300)
    plt.close()

    print("=" * 50 + "\n")

    # ==========================================
    # 3. Run online Q-Learning optimization loop
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

        # (a) Map continuous channel state to discrete state index
        s_tuple = discretizer.discretize(h)
        s_next_tuple = discretizer.discretize(h_next)

        # (b) Action selection (Epsilon-Greedy over 2^N Discrete Actions)
        action_idx, binary_action = agent.select_action(s_tuple)

        # (c) Calculate physical layer Rate for current action (Immediate Reward)
        reward = bisection(h / 1000000, binary_action)[0]

        # (d) Q-Table tabular update
        agent.learn(s_tuple, action_idx, reward, s_next_tuple)

        # (e) Record performance metrics
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

    # 4. Save data to ./result-QLearning/
    save_data(rate_his_ratio, os.path.join(results_dir, "normalized_computation_rate.txt"))
    save_data(throughput_his, os.path.join(results_dir, "throughput_mbps.txt"))
    save_data(latency_his, os.path.join(results_dir, "latency_ms.txt"))
    save_data(agent.cost_his, os.path.join(results_dir, "cost_his.txt"))

    np.save(os.path.join(results_dir, "normalized_computation_rate.npy"), np.array(rate_his_ratio))
    np.save(os.path.join(results_dir, "throughput_mbps.npy"), np.array(throughput_his))
    np.save(os.path.join(results_dir, "latency_ms.npy"), np.array(latency_his))

    # 5. Plot and save curves
    agent.plot_cost(save_path=os.path.join(results_dir, "qlearning_loss_convergence.png"))
    plot_normalized_rate(rate_his_ratio, save_path=os.path.join(results_dir, "normalized_computation_rate.png"))
    plot_throughput(throughput_his, save_path=os.path.join(results_dir, "throughput.png"))
    plot_cdf_latency(latency_his, save_path=os.path.join(results_dir, "cdf_of_latency.png"))

    # 6. Real video transmission time evaluation
    calculate_video_transmission_time(task_folder="./task",
                                      avg_normalized_rate=avg_normalized_rate,
                                      results_dir=results_dir)