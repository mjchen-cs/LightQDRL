## LightQDRL

# LightQDRL: Lightweight Quantum Deep Reinforcement Learning for Real-Time Edge Intelligence Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-purple.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LightQDRL** is a hybrid Quantum-Classical Deep Reinforcement Learning framework designed for real-time binary offloading and resource allocation in 5G/B5G wireless edge networks. By integrating **Parameterized Quantum Circuits (PQCs)** with **Classical Deep Neural Networks (HQNN)**, LightQDRL significantly reduces trainable parameter footprints while maintaining near-optimal offloading throughput and latency performance.

---

## 🌟 Key Features

* **Hybrid Quantum-Classical Architecture (HQNN)**: Combines classical 1D neural layers for state encoding/decoding with PennyLane-based Variational Quantum Circuits (VQCs) for high-dimensional feature representation.
* **Order-Preserving Quantization**: An efficient action quantization strategy to map continuous Q-value outputs to binary offloading decisions ($m \in \{0, 1\}^N$).
* **Quantum Gradient Regularization**: Implements gradient clipping during backpropagation to mitigate Barren Plateaus commonly found in VQC optimization.
* **5G Network Simulation & Evaluation**: Includes realistic 5G physical layer modeling (Bisection method) and real-world video transmission latency estimation.
* **Automated Result Export**: Generates high-resolution performance plots (Loss Convergence, Normalized Computation Rate, System Throughput, and Latency CDF) and saves experiment metrics automatically.

---

## 🏗 System Architecture

The core framework consists of three main stages:

[ Channel State h ] ──> [ Classical Encoder ] ──> [ Angle Encoding [-π, π] ]
│
▼
[ Binary Offloading Action m ] <── [ Classical Decoder ] <── [ PennyLane VQC (RX, RY, RZ + CNOT) ]
│
▼
[ Physical Layer Bisection Optimization ] ──> [ Experience Replay & Policy Update ]

1. **State Encoding**: Channel gains $h$ are normalized and mapped to quantum rotation angles within $[-\pi, \pi]$.
2. **Variational Quantum Circuit (VQC)**: Multi-layer entangled quantum ansatz ($R_X, R_Y, R_Z$ gates with entangling CNOT rings) measures Pauli-Z expectation values.
3. **Action Decoding & Optimization**: Features are converted to continuous Q-values, decoded via Order-Preserving Quantization, and evaluated via physical layer bisection optimization.

---

## 📁 Repository Structure
├── data/                       # Dataset directory (.mat files containing channel state matrices)
│   └── data_10.mat             # Example dataset for N=10 users
├── task/                       # Test folder containing sample .mp4 files for 5G transmission testing
│   └── sample_video.mp4
├── results/                    # Output folder for metrics, figures, and prediction reports
│   ├── normalized_computation_rate.png
│   ├── throughput.png
│   ├── cdf_of_latency.png
│   ├── quantum_loss_convergence.png
│   └── transmission_results.txt
├── optimization.py             # Physical layer bisection optimization algorithm
├── LightQDRL.py               # Main execution script for LightQDRL
└── README.md                   # Project documentation



---

## 🛠 Prerequisites & Installation

### Requirements

* Python 3.8+
* PyTorch 1.12+
* PennyLane 0.30+
* SciPy, NumPy, Pandas, Matplotlib

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/LightQDRL.git](https://github.com/your-username/LightQDRL.git)
   cd LightQDRL
### Install dependencies: 
```bash
   pip install torch pennylane scipy numpy pandas matplotlib
   

## 🚀 Running the Model
1. Prepare DataEnsure that the dataset for $N$ users is placed in the ./data/ directory. For example, for $N=10$, data_10.mat should exist in ./data/.
2. Run Main Training and EvaluationExecute the main script to start the online optimization loop, save metrics, generate plots, and assess video transmission time:

```bash
python ResNet-QDRL.py


⚙️ Hyperparameters Configuration
Key parameters can be adjusted directly inside ResNet-QDRL.py

N = 10                       # Number of users / channels
n = 30000                    # Total number of time frames
n_qubits = 6                 # Number of quantum qubits (4 ~ 8)
n_layers = 2                 # Depth of Quantum Circuit Ansatz
Memory = 1024                # Experience Replay Memory size
REAL_5G_BASE_RATE_MBPS = 300 # Base 5G rate benchmark (Mbps)
TASK_SIZE_BITS = 5MB * 8     # Edge task size benchmark

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
