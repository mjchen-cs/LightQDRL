import scipy.io as sio  # import scipy.io for .mat file I/O
import numpy as np  # import numpy
import torch
import random
import time
import os  # Used for folder operations and getting file sizes
import glob  # Used for matching video file names
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# Retain physical layer bisection optimization algorithm
from optimization import bisection

# Set random seed to ensure experiment reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ==========================================
# 1. State Discretization Module (State Discretizer)
# ==========================================
class StateDiscretizer:
    """
    Maps continuous channel gain vector h (dimension N) to a finite discrete state (Tuple)
    Used to establish state indexing for quantum Q-Learning
    """

    def __init__(self, n_users, n_bins=5):
        self.n_users = n_users
        self.n_bins = n_bins
        self.bins = []

    def fit(self, channel_data):
        """Dynamically fit bin boundaries based on overall channel distribution"""
        self.bins = []
        for dim in range(self.n_users):
            dim_data = channel_data[:, dim]
            b = np.linspace(dim_data.min(), dim_data.max(), self.n_bins + 1)[1:-1]
            self.bins.append(b)

    def discretize(self, h):
        """Map a single channel state h to a discrete state tuple"""
        discrete_s = []
        for dim in range(self.n_users):
            idx = np.digitize(h[dim], self.bins[dim])
            discrete_s.append(idx)
        return tuple(discrete_s)


# ==========================================
# 2. Quantum Q-Learning Agent
# ==========================================
class QuantumQLearningAgent:
    """
    Quantum Q-Learning agent (Quantum-inspired Q-Learning)
    - Core mechanism: Uses quantum superposition state |psi> = cos(theta)|0> + sin(theta)|1>
    - Measurement mechanism: Selects binary action vector based on probability collapse P(|1>) = sin^2(theta)
    - Update mechanism: Uses TD-Error to control quantum rotation gate rotation phase angle theta
    - Advantage: Reduces action space indexing complexity from O(2^N) in traditional Q-tables to O(N), avoiding the curse of dimensionality
    """

    def __init__(self, n_users, learning_rate=0.05, gamma=0.9, theta_step=0.02):
        self.n_users = n_users
        self.lr = learning_rate
        self.gamma = gamma
        self.theta_step = theta_step

        # Quantum theta table: state_tuple -> theta_array (dimension N)
        # Initial angle set to pi/4 (i.e., cos^2 = sin^2 = 0.5, maximum quantum superposition state and uniform exploration)
        self.theta_table = {}
        # Sparse Q-value table, used to calculate Bellman target values
        self.q_table = {}
        self.cost_his = []  # Record squared TD Error as convergence loss

    def get_angles(self, state_tuple):
        """Find or initialize the quantum phase angle vector theta for the current state"""
        if state_tuple not in self.theta_table:
            self.theta_table[state_tuple] = np.full(self.n_users, np.pi / 4.0)
        return self.theta_table[state_tuple]

    def get_q_dict(self, state_tuple):
        """Get the sparse Q-dictionary for a state with binary action tuples as keys"""
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {}
        return self.q_table[state_tuple]

    def select_action(self, state_tuple):
        """
        Quantum collapse measurement (Quantum Measurement):
        Perform measurement sampling according to probability P(|1>) = sin^2(theta) to obtain an N-dimensional binary action vector
        """
        angles = self.get_angles(state_tuple)
        prob_1 = np.sin(angles) ** 2  # Quantum probability of measuring |1>

        # Quantum measurement collapse
        rand_vals = np.random.rand(self.n_users)
        binary_action = (rand_vals < prob_1).astype(int)

        return tuple(binary_action), binary_action, prob_1

    def learn(self, s_tuple, binary_action, reward, s_next_tuple):
        """
        TD-Error based quantum rotation gate probability update algorithm
        """
        angles = self.get_angles(s_tuple)
        q_dict = self.get_q_dict(s_tuple)
        q_next_dict = self.get_q_dict(s_next_tuple)

        act_key = tuple(binary_action)

        # Get current predicted Q-value and maximum predicted Q-value of the next state
        predict_q = q_dict.get(act_key, reward)
        max_next_q = max(q_next_dict.values()) if len(q_next_dict) > 0 else 0.0
        target_q = reward + self.gamma * max_next_q

        td_error = target_q - predict_q

        # 1. Update sparse Q-table
        q_dict[act_key] = predict_q + self.lr * td_error
        self.cost_his.append(td_error ** 2)

        # 2. Quantum rotation gate update mechanism:
        # Calculate rotation gradient direction: (a_i - P(|1>))
        prob_1 = np.sin(angles) ** 2
        grad_direction = binary_action - prob_1

        # Calculate rotation angle delta_theta
        delta_theta = self.theta_step * np.tanh(td_error) * grad_direction
        updated_angles = angles + delta_theta

        # Boundary limits to prevent oversaturation causing loss of exploration [eps, pi/2 - eps]
        eps = 0.01
        self.theta_table[s_tuple] = np.clip(updated_angles, eps, np.pi / 2.0 - eps)

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_his, color='purple', linewidth=1.2, alpha=0.7)
        plt.ylabel('TD Error Squared')
        plt.xlabel('Learning Steps')
        plt.title('Convergence Curve of Quantum Q-Learning (QQL)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


# ==========================================
# 3. Data Saving and Plotting Utility Functions
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
    plt.title('Normalized Computation Rate vs. Time Frames (Quantum Q-Learning)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_throughput(throughput_his, n_users, save_path=None):
    """
    Plot dual Y-axis throughput curves:
    - System Total Throughput
    - Per-User Average Throughput
    """
    tp_array = np.asarray(throughput_his)
    cum_sys_tp = np.cumsum(tp_array) / (np.arange(len(tp_array)) + 1)
    cum_user_tp = cum_sys_tp / n_users

    fig, ax1 = plt.subplots(figsize=(8, 5))
    frames = np.arange(len(tp_array)) + 1

    color = '#2ca02c'
    ax1.set_xlabel('Time Frames')
    ax1.set_ylabel('System Total Throughput (Mbps)', color=color)
    line1 = ax1.plot(frames, cum_sys_tp, color=color, label='System Total Throughput', linewidth=2.0)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Per-User Average Throughput (Mbps)', color=color)
    line2 = ax2.plot(frames, cum_user_tp, color=color, linestyle='--', label='Per-User Average Throughput',
                     linewidth=2.0)
    ax2.tick_params(axis='y', labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('System Total vs. Per-User Average Throughput (Quantum Q-Learning)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cdf_latency(latency_his, save_path=None):
    sorted_latency = np.sort(np.asarray(latency_his))
    cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(sorted_latency, cdf, color='#d62728', linewidth=2.0, label='Quantum Q-Learning Latency CDF')

    plt.ylabel('Cumulative Probability (CDF)')
    plt.xlabel('Transmission & Processing Latency (ms)')
    plt.title('Cumulative Distribution Function (CDF) of Latency (Quantum Q-Learning)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def calculate_video_transmission_time(task_folder, avg_sys_throughput_mbps, avg_user_throughput_mbps,
                                      results_dir="./result-QuantumQLearning"):
    actual_avg_rate_bps = avg_sys_throughput_mbps * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (Quantum Q-Learning) ---")
    print(f"Quantum Q-Learning predicted real 5G system total throughput: {avg_sys_throughput_mbps:.3f} Mbps")
    print(f"Quantum Q-Learning predicted real 5G per-user throughput: {avg_user_throughput_mbps:.3f} Mbps")

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
        f.write("    Quantum Q-Learning Evaluation Report\n")
        f.write("=========================================================\n")
        f.write(f"System Average Total Throughput: {avg_sys_throughput_mbps:.3f} Mbps\n")
        f.write(f"Per-User Average Throughput: {avg_user_throughput_mbps:.3f} Mbps\n")
        f.write(f"Total Processed Video Files: {len(video_files)}\n")
        f.write(f"Total Video File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} seconds\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")
        f.write("=========================================================\n")

    print(f"Video transmission time analysis exported to: '{output_file}'\n")


# ==========================================
# 4. Main Program Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Create independent result storage directory ./result-QuantumQLearning
    results_dir = "./result-QuantumQLearning"
    os.makedirs(results_dir, exist_ok=True)

    N = 10  # Number of users
    n = 30000  # Number of time frames
    SINGLE_USER_BASE_RATE_MBPS = 60.0  # Single-user baseline rate 60 Mbps
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # Edge task size (5 MB)

    print('#user = %d, #channel = %d (Quantum Q-Learning Benchmarking)' % (N, n))

    # 1. Load channel data and baseline rate
    channel = sio.loadmat('./data/data_%d' % N)['input_h']
    rate = sio.loadmat('./data/data_%d' % N)['output_obj']
    channel = channel * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    # 2. Fit state discretizer and initialize quantum agent
    discretizer = StateDiscretizer(n_users=N, n_bins=5)
    discretizer.fit(channel)

    # Instantiate Quantum Q-Learning agent
    agent = QuantumQLearningAgent(n_users=N, learning_rate=0.05, gamma=0.9, theta_step=0.02)

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    throughput_his = []
    latency_his = []

    # 3. Run online Quantum Q-Learning optimization loop
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

        # (b) Quantum measurement action selection (based on probability amplitude collapse)
        act_key, binary_action, prob_1 = agent.select_action(s_tuple)

        # (c) Calculate physical layer Rate of current action (i.e., immediate reward)
        reward = bisection(h / 1000000, binary_action)[0]

        # (d) Update phase angle based on quantum rotation gate
        agent.learn(s_tuple, binary_action, reward, s_next_tuple)

        # (e) Record performance metrics
        norm_rate = reward / rate[i_idx][0]

        # System total baseline bandwidth dynamically scales with the number of users N (multi-user concurrent physical scenario)
        current_sys_bandwidth = SINGLE_USER_BASE_RATE_MBPS * N
        current_throughput_mbps = norm_rate * current_sys_bandwidth
        current_latency_ms = (TASK_SIZE_BITS / (current_throughput_mbps * 1e6)) * 1000.0

        rate_his.append(reward)
        rate_his_ratio.append(norm_rate)
        throughput_his.append(current_throughput_mbps)
        latency_his.append(current_latency_ms)

    total_time = time.time() - start_time

    # Calculate average metrics during convergence/testing phase
    avg_normalized_rate = sum(rate_his_ratio[-num_test:]) / num_test
    avg_sys_throughput_mbps = sum(throughput_his[-num_test:]) / num_test
    avg_per_user_throughput_mbps = avg_sys_throughput_mbps / N  # Per-user average throughput

    print("\n==================== Quantum Q-Learning Performance Evaluation Report ====================")
    print("Averaged normalized computation rate  :", avg_normalized_rate)
    print(f"System Total Throughput               : {avg_sys_throughput_mbps:.4f} Mbps (Increases with user count N)")
    print(f"Per-User Throughput                   : {avg_per_user_throughput_mbps:.4f} Mbps (Decreases with user count N)")
    print('Total time consumed                   : %s' % total_time)
    print('Average time per channel              : %s' % (total_time / n))
    print("=====================================================================\n")

    # 4. Save data to ./result-QuantumQLearning/
    save_data(rate_his_ratio, os.path.join(results_dir, "normalized_computation_rate.txt"))
    save_data(throughput_his, os.path.join(results_dir, "throughput_mbps.txt"))
    save_data(latency_his, os.path.join(results_dir, "latency_ms.txt"))
    save_data(agent.cost_his, os.path.join(results_dir, "cost_his.txt"))

    # Save text record of system total throughput and per-user average throughput
    with open(os.path.join(results_dir, "average_throughput_mbps.txt"), 'w') as f:
        f.write(f"System Total Throughput: {avg_sys_throughput_mbps:.4f} Mbps\n")
        f.write(f"Per-User Average Throughput: {avg_per_user_throughput_mbps:.4f} Mbps\n")

    np.save(os.path.join(results_dir, "normalized_computation_rate.npy"), np.array(rate_his_ratio))
    np.save(os.path.join(results_dir, "throughput_mbps.npy"), np.array(throughput_his))
    np.save(os.path.join(results_dir, "latency_ms.npy"), np.array(latency_his))

    # 5. Plot and save curves
    agent.plot_cost(save_path=os.path.join(results_dir, "qlearning_loss_convergence.png"))
    plot_normalized_rate(rate_his_ratio, save_path=os.path.join(results_dir, "normalized_computation_rate.png"))
    plot_throughput(throughput_his, n_users=N, save_path=os.path.join(results_dir, "throughput.png"))
    plot_cdf_latency(latency_his, save_path=os.path.join(results_dir, "cdf_of_latency.png"))

    # 6. Real video transmission latency evaluation
    calculate_video_transmission_time(task_folder="./task",
                                      avg_sys_throughput_mbps=avg_sys_throughput_mbps,
                                      avg_user_throughput_mbps=avg_per_user_throughput_mbps,
                                      results_dir=results_dir)