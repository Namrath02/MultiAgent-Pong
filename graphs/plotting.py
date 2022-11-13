import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


f = open("reward.txt", "r")
# read alternate lines into two different lists
r1 = []
r2 = []

for i, line in enumerate(f):
    if i % 2 == 0:
        r1.append(float(line))
    else:
        r2.append(float(line))

f.close()


# # plot the rewards in a graph
plt.plot(r1, label="Player 1", color="red")
plt.plot(r2, label="Player 2", color="blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards of Competitive Training")
plt.legend()
plt.show()
