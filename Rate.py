import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl




mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 1.5

# Ensure that PDF text is editable
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# ==========================================================
# Load data
# ==========================================================

file = "Rate.xlsx"

df = pd.read_excel(file)


# Delete possible spaces in column names
df.columns = df.columns.str.strip()


print(df.columns)
print(df.head())


# ==========================================================
# Parameters
# ==========================================================

episode = df["Episode"]


algorithms = [
    "Quantum Q-Learning",
    "Quantum-DQN",
    "Quantum-PPO",
    "LightQDRL"
]


# ==========================================================
# Average every 3000 episodes
# 30000 episodes -> 10 points
# ==========================================================

window = 3000


avg_df = df.groupby(
    np.arange(len(df)) // window
).mean()


# Calculate the average
avg_episode = avg_df["Episode"]



# ==========================================================
# Plot
# ==========================================================

fig = plt.figure(
    figsize=(7, 6),
    dpi=600
)

ax = plt.gca()



# Nature color style

colors = {
    "Quantum Q-Learning":"#7F7F7F",
    "Quantum-DQN":"#55A868",
    "Quantum-PPO":"#DD8452",
    "LightQDRL":"#1F77B4"
}


markers = {
    "Quantum Q-Learning":"s",
    "Quantum-DQN":"^",
    "Quantum-PPO":"D",
    "LightQDRL":"o"
}



# ==========================================================
# Draw four algorithms
# ==========================================================

for alg in algorithms:


    if alg == "LightQDRL":

        ax.plot(
            avg_episode,
            avg_df[alg],
            color=colors[alg],
            linewidth=3.5,
            marker=markers[alg],
            markersize=6,
            label=alg
        )

    else:

        ax.plot(
            avg_episode,
            avg_df[alg],
            color=colors[alg],
            linewidth=3,
            marker=markers[alg],
            markersize=5,
            label=alg
        )



# ==========================================================
# Axis
# ==========================================================

ax.set_xlabel(
    "Training Episode"
)

ax.set_ylabel(
    "Normalized Computation Rate"
)


ax.set_xlim(
    0,
    30000
)



ax.set_ylim(
    0.5,
    1.05
)






# remove top/right frame

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



ax.tick_params(
    direction="out",
    length=3,
    width=1
)


ax.grid(
    linestyle="--",
    linewidth=0.5,
    alpha=0.3
)



ax.legend(
    frameon=False,
    fontsize=16,
    loc="lower right"
)



plt.tight_layout()



# ==========================================================
# Save
# ==========================================================

plt.savefig(
    "Fig3a_Normalized_Computation_Rate.pdf",
    bbox_inches="tight"
)


plt.savefig(
    "Fig3a_Normalized_Computation_Rate.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()

# ==========================================================
# Calculate Final mean + 95% CI
# ==========================================================

from scipy import stats


# selected 3000 Episode
# Episode 27000-30000

final_df = df[df["Episode"] >= 27000]


print("\n==============================")
print("Final Performance (Mean ± 95% CI)")
print("==============================")


for alg in algorithms:

    data = final_df[alg].values


    # sample number
    N = len(data)


    # mean
    mean_value = np.mean(data)


    # standard deviation
    std_value = np.std(data, ddof=1)


    # standard error
    SE = std_value / np.sqrt(N)


    # 95% confidence interval
    CI = 1.96 * SE


    lower = mean_value - CI
    upper = mean_value + CI


    print(
        f"{alg}: "
        f"{mean_value:.4f} ± {CI:.4f}"
    )

    print(
        f"95% CI range: "
        f"[{lower:.4f}, {upper:.4f}]"
    )
