import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl




mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.linewidth'] = 1.0

# PDF文字可编辑
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



# ==========================================================
# Load throughput data
# ==========================================================

file = "throughput.xlsx"


df = pd.read_excel(file)




df.columns = df.columns.str.strip()


print(df.columns)

print(df.head())



# ==========================================================
# Algorithm names
# ==========================================================

algorithms = [

    "Traditional Quantum Q-Learning",

    "Traditional Quantum-DQN",

    "Traditional Quantum-PPO",

    "LightQDRL"

]


# ==========================================================
# Average every 3000 episodes
# 30000 -> 10 points
# ==========================================================

window = 3000


avg_df = df.groupby(
    np.arange(len(df)) // window
).mean()



avg_episode = avg_df["Episode"]



# ==========================================================
# Calculate Final mean + 95% CI
# Last 3000 episodes
# ==========================================================


final_df = df[df["Episode"] >= 27000]


print("\n================================")
print("Final Throughput Performance")
print("(Mean ± 95% CI)")
print("================================")


final_results = []


for alg in algorithms:


    data = final_df[alg].values


    N = len(data)


    mean_value = np.mean(data)


    std_value = np.std(
        data,
        ddof=1
    )


    SE = std_value / np.sqrt(N)


    CI95 = 1.96 * SE


    final_results.append(
        [
            alg,
            mean_value,
            CI95
        ]
    )


    print(
        f"{alg}: "
        f"{mean_value:.3f} ± {CI95:.3f} Mbps"
    )



# save

result_df = pd.DataFrame(
    final_results,
    columns=[
        "Algorithm",
        "Final Mean Throughput(Mbps)",
        "95% CI"
    ]
)


result_df.to_excel(
    "Final_Throughput_CI.xlsx",
    index=False
)



# ==========================================================
# Plot Fig.3a style
# ==========================================================


fig = plt.figure(
    figsize=(3.5,2.8),
    dpi=600
)


ax = plt.gca()



# Nature style colors

colors = {

    "Traditional Quantum Q-Learning":
    "#7F7F7F",

    "Traditional Quantum-DQN":
    "#55A868",

    "Traditional Quantum-PPO":
    "#DD8452",

    "LightQDRL":
    "#1F77B4"

}



markers = {

    "Traditional Quantum Q-Learning":
    "s",

    "Traditional Quantum-DQN":
    "^",

    "Traditional Quantum-PPO":
    "D",

    "LightQDRL":
    "o"

}



# ==========================================================
# Plot four algorithms
# ==========================================================


for alg in algorithms:


    if alg == "LightQDRL":


        ax.plot(

            avg_episode,

            avg_df[alg],

            color=colors[alg],

            linewidth=2.5,

            marker=markers[alg],

            markersize=6,

            label=alg

        )


    else:


        ax.plot(

            avg_episode,

            avg_df[alg],

            color=colors[alg],

            linewidth=1.8,

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
    "Throughput (Mbps)"
)


ax.set_xlim(
    0,
    30000
)




y_min = avg_df[algorithms].min().min()

y_max = avg_df[algorithms].max().max()


ax.set_ylim(
    y_min*0.95,
    y_max*1.05
)






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
    fontsize=6.5,
    loc="best"
)



plt.tight_layout()



# ==========================================================
# Save
# ==========================================================


plt.savefig(

    "Fig3b_Throughput.pdf",

    bbox_inches="tight"

)


plt.savefig(

    "Fig3b_Throughput.png",

    dpi=600,

    bbox_inches="tight"

)



plt.show()