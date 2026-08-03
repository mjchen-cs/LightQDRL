import scipy.io as sio  # import scipy.io for .mat file I/O
import numpy as np  # import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import random
import time
import os  # Used for folder operations and getting file sizes
import glob  # Used for matching video file names
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# ==========================================
# Core Import: PennyLane Variational Quantum Circuit (VQC) Library
# ==========================================
import pennylane as qml

# Retain physical layer bisection optimization algorithm
from optimization import bisection

# Set random seed to ensure reproducibility of quantum and RL experiments
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==========================================
# 1. Paper-Standard Variational Quantum Circuit (VQC) Architecture
# ==========================================
n_qubits = 6  # Keep the number of qubits consistent with the system
n_layers = 2  # Keep the circuit depth

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Variational quantum circuit architecture shared by traditional Q-PPO
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
# 2. Traditional Quantum Actor-Critic Network
# ==========================================
class QuantumActorCritic(nn.Module):
    """
    Quantum Actor-Critic Joint Network:
    Uses a shared VQC to extract channel features, Actor outputs action distribution, Critic outputs state value V(s)
    """

    def __init__(self, n_users, n_qubits=6):
        super(QuantumActorCritic, self).__init__()
        self.n_users = n_users
        self.n_qubits = n_qubits

        # Classical feature encoder: Use LayerNorm to avoid training mode errors when Batch Size = 1
        self.encoder = nn.Sequential(
            nn.Linear(n_users, 64),
            nn.LayerNorm(64),  # BatchNorm1d has been replaced by LayerNorm
            nn.ReLU(),
            nn.Linear(64, self.n_qubits),
            nn.Tanh()
        )

        # Variational quantum layer
        weight_shapes = {"weights": (n_layers, self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Actor decoding head (outputs action logits for N users)
        self.actor_head = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_users)
        )

        # Critic decoding head (outputs state value V(s))
        self.critic_head = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x_angles = self.encoder(x) * np.pi
        q_features = self.q_layer(x_angles)
        action_logits = self.actor_head(q_features)
        state_value = self.critic_head(q_features)
        return action_logits, state_value

    def get_action_and_value(self, state, action=None):
        """Generate Multi-Bernoulli probability distribution, and compute Action, Log Prob, Entropy, and Value"""
        logits, value = self.forward(state)
        dist = Bernoulli(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action, log_prob, entropy, value


# ==========================================
# 3. Traditional Quantum-PPO Agent
# ==========================================
class TraditionalQPPOAgent:
    """
    Traditional Quantum PPO Agent (with GAE and PPO-Clip surrogate loss function)
    """

    def __init__(self, n_users, lr=0.003, gamma=0.9, gae_lambda=0.95, clip_eps=0.2, ppo_epochs=4, batch_size=64, update_interval=128):
        self.n_users = n_users
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.update_interval = update_interval

        self.ac_net = QuantumActorCritic(n_users, n_qubits)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

        # Trajectory Buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

        self.cost_his = []

    def select_action(self, state):
        """Sample actions in online evaluation mode"""
        self.ac_net.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = self.ac_net.get_action_and_value(state_tensor)

        action_np = action.squeeze(0).cpu().numpy().astype(int)
        log_prob_val = log_prob.item()
        value_val = value.item()

        return action_np, log_prob_val, value_val

    def store_transition(self, s, a, r, log_prob, val, done=False):
        """Store transitions into PPO trajectory buffer"""
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.log_probs.append(log_prob)
        self.values.append(val)
        self.dones.append(done)

    def learn(self, last_state):
        """Trigger PPO off-policy updates when trajectory count reaches update_interval"""
        if len(self.states) < self.update_interval:
            return

        # 1. Compute Value of the last state in eval mode (avoiding single-sample dimension errors)
        self.ac_net.eval()
        last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_val = self.ac_net.get_action_and_value(last_state_tensor)
            last_val = last_val.item()

        # 2. Compute GAE advantage function and Returns
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_val])
        dones = np.array(self.dones + [False])

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam

        returns = advantages + values[:-1]

        # Convert to Tensor format
        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.FloatTensor(np.array(self.actions))
        old_log_probs_t = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).unsqueeze(1)
        returns_t = torch.FloatTensor(returns).unsqueeze(1)

        # Advantage normalization
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # 3. Switch to train mode, enable PPO multi-Epoch Batch gradient updates
        self.ac_net.train()
        dataset_size = len(self.states)
        total_loss_epoch = 0.0

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = permutation[start_idx:start_idx + self.batch_size]

                b_states = states_t[batch_indices]
                b_actions = actions_t[batch_indices]
                b_old_log_probs = old_log_probs_t[batch_indices]
                b_advantages = advantages_t[batch_indices]
                b_returns = returns_t[batch_indices]

                _, new_log_probs, entropy, new_values = self.ac_net.get_action_and_value(b_states, b_actions)

                # Ratio
                ratios = torch.exp(new_log_probs - b_old_log_probs)

                # Clipped Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(new_values, b_returns)

                # Entropy Loss
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
                self.optimizer.step()

                total_loss_epoch += loss.item()

        self.cost_his.append(total_loss_epoch / self.ppo_epochs)

        # Clear Buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_his, color='purple', linewidth=1.5)
        plt.ylabel('Traditional Q-PPO Loss')
        plt.xlabel('PPO Update Iterations')
        plt.title('Convergence Curve of Traditional Quantum-PPO')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


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
    plt.title('Normalized Computation Rate vs. Time Frames (Traditional Q-PPO)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


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
    plt.title('System Throughput over Time Frames (Traditional Q-PPO)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cdf_latency(latency_his, save_path=None):
    sorted_latency = np.sort(np.asarray(latency_his))
    cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(sorted_latency, cdf, color='#d62728', linewidth=2.0, label='Traditional Q-PPO Latency CDF')

    plt.ylabel('Cumulative Probability (CDF)')
    plt.xlabel('Transmission & Processing Latency (ms)')
    plt.title('Cumulative Distribution Function (CDF) of Latency (Traditional Q-PPO)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def calculate_video_transmission_time(task_folder, avg_normalized_rate, results_dir="./result-QPPO"):
    REAL_5G_BASE_RATE_MBPS = 300.0
    actual_avg_rate_bps = avg_normalized_rate * REAL_5G_BASE_RATE_MBPS * 1e6

    print(f"\n--- Physical Layer Synchronization Evaluation (Traditional Q-PPO) ---")
    print(f"Traditional Q-PPO normalized efficiency: {avg_normalized_rate:.4f}")
    print(f"Predicted real 5G average transmission rate: {actual_avg_rate_bps / 1e6:.3f} Mbps")

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
        f.write("    Traditional Quantum-PPO Evaluation Report\n")
        f.write("=========================================================\n")
        f.write(f"5G Network Baseline Setting: {REAL_5G_BASE_RATE_MBPS:.1f} Mbps\n")
        f.write(f"Optimized Transmission Rate: {actual_avg_rate_bps / 1e6:.3f} Mbps\n")
        f.write(f"Number of Processed Video Files: {len(video_files)}\n")
        f.write(f"Total File Size: {total_size_bytes / (1024 * 1024):.2f} MB\n")
        f.write(f"Estimated Total Transmission Time: {total_time_seconds:.3f} seconds\n")
        f.write("---------------------------------------------------------\n")
        for res in results:
            f.write(f"{res['name']:<30} | {res['size_mb']:<15.2f} | {res['time_sec']:<15.3f}\n")
        f.write("=========================================================\n")

    print(f"Transmission time analysis exported to: '{output_file}'\n")


# ==========================================
# 5. Main Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Create independent result storage directory ./result-QPPO
    results_dir = "./result-QPPO"
    os.makedirs(results_dir, exist_ok=True)

    N = 10  # Number of users
    n = 30000  # Number of frames
    REAL_5G_BASE_RATE_MBPS = 300.0  # 5G baseline bandwidth
    TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8  # 5MB task size

    print('#user = %d, #channel = %d (Traditional Quantum-PPO Benchmarking)' % (N, n))

    # 1. Load channel and baseline Rate
    channel = sio.loadmat('./data/data_%d' % N)['input_h']
    rate = sio.loadmat('./data/data_%d' % N)['output_obj']
    channel = channel * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    # 2. Instantiate Traditional Quantum-PPO Agent
    agent = TraditionalQPPOAgent(n_users=N, lr=0.003, update_interval=128)

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    throughput_his = []
    latency_his = []

    # 3. Run online RL loop
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

        # (a) Action sampling and state value estimation (Multi-Bernoulli Action Sampling)
        binary_action, log_prob, val = agent.select_action(h)

        # (b) Calculate physical layer Rate of current action (i.e. Immediate Reward)
        reward = bisection(h / 1000000, binary_action)[0]

        # (c) Store transitions and perform PPO gradient updates
        agent.store_transition(h, binary_action, reward, log_prob, val)
        agent.learn(h_next)

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

    # 4. Save data to ./result-QPPO/
    save_data(rate_his_ratio, os.path.join(results_dir, "normalized_computation_rate.txt"))
    save_data(throughput_his, os.path.join(results_dir, "throughput_mbps.txt"))
    save_data(latency_his, os.path.join(results_dir, "latency_ms.txt"))
    save_data(agent.cost_his, os.path.join(results_dir, "cost_his.txt"))

    np.save(os.path.join(results_dir, "normalized_computation_rate.npy"), np.array(rate_his_ratio))
    np.save(os.path.join(results_dir, "throughput_mbps.npy"), np.array(throughput_his))
    np.save(os.path.join(results_dir, "latency_ms.npy"), np.array(latency_his))

    # 5. Plot and save curves
    agent.plot_cost(save_path=os.path.join(results_dir, "qppo_loss_convergence.png"))
    plot_normalized_rate(rate_his_ratio, save_path=os.path.join(results_dir, "normalized_computation_rate.png"))
    plot_throughput(throughput_his, save_path=os.path.join(results_dir, "throughput.png"))
    plot_cdf_latency(latency_his, save_path=os.path.join(results_dir, "cdf_of_latency.png"))

    # 6. Real video transmission time test
    calculate_video_transmission_time(task_folder="./task",
                                      avg_normalized_rate=avg_normalized_rate,
                                      results_dir=results_dir)

    # rate: 0.73182350116332
    # Total time  consumed: 354.5754506587982
    # Average time  per channel: 0.011819181688626607