import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([100, 1000, 20000], [20, 50, 100.90], label = "scikit");  # Plot some data on the axes.
ax.plot([1, 2, 3, 4], [1, 3, 2, 5]);  # Plot some data on the axes.


plt.savefig("test.png")