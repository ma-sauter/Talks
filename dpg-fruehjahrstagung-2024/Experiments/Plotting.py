import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import znnl as nl


# load data
lossnames = ["CrossEntropy", "LPNorm2", "MeanPowerLoss2"]
path = "C:/Users/Admin/Uni/Talks/dpg-fruehjahrstagung-2024/Experiments/"
recordernames = ["l_train_recorder_", "r_train_recorder_", "test_recorder_"]
recorders = dict()
reports = dict()

for lossname in lossnames:
    print("Loading " + lossname + "...")
    for recordername in recordernames:
        recorders[recordername + lossname] = nl.training_recording.DataStorage(
            path + recordername + lossname
        )
        if recordername[:4] == "test":
            reports[recordername + lossname] = recorders[
                recordername + lossname
            ].fetch_data(["loss", "accuracy"])
        else:
            reports[recordername + lossname] = recorders[
                recordername + lossname
            ].fetch_data(
                ["loss", "accuracy", "eigenvalues", "entropy", "covariance_entropy"]
            )


lossname = lossnames[0]
N_eigenvalues = 10

# Create figure and axes
fig, ax = plt.subplots()

# Initialize empty lines for each recorder
lines = []

for i in range(N_eigenvalues):
    lines.append(ax.plot([], [], "o", color="orange")[0])
for i in range(N_eigenvalues):
    lines.append(ax.plot([], [], ".", color="blue")[0])


# Set up the animation function
def animate(frame):
    # Update the data for each line
    data_r = reports["r_train_recorder_" + lossname]["eigenvalues"]
    data_l = reports["l_train_recorder_" + lossname]["eigenvalues"]
    for i in range(N_eigenvalues):
        x = i * 5
        y1 = data_r[frame][i]
        y2 = data_l[frame][i]
        lines[i].set_data(x, y1)
        lines[2 * i].set_data(x, y2)

    # Set the x and y limits of the plot
    ax.set_xlim(0, N_eigenvalues * 5)
    ax.set_ylim(0, np.max(np.max(data_r), np.max(data_l)))

    # Set the title and legend
    ax.set_title("Eigenvalue Animation")
    ax.legend()


# Create the animation
animation = FuncAnimation(
    fig,
    animate,
    frames=len(reports["r_train_recorder_" + lossname]["eigenvalues"]),
    interval=1 / 60 * 1000,  # 60 fps
    blit=True,
)

# Show the plot
plt.show()
