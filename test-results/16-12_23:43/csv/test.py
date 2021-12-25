import matplotlib.pyplot as plt
import pandas as pd



times = pd.read_csv("times.csv")
times.set_index("samples")
print(times.head())

times.plot()
plt.savefig("test.png")