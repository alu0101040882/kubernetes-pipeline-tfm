
import matplotlib.pyplot as plt
import pandas as pd


import os

testName = "26-12_11:22"

workdir = f"{os.path.dirname(os.path.realpath(__file__))}/{testName}"


speedups = pd.read_csv(f"{workdir}/csv/speedup.csv")
times = pd.read_csv(f"{workdir}/csv/times.csv")

print(times)
print(speedups)

test_samples = speedups["Samples"]

y_labels = []
for sample in test_samples:
    y_labels.append(str(sample))

scikitTimes = times["Scikit"]
kubeTimes = times["Kubernetes"]

speedUps = speedups["Speed Up"]

plt.figure()
plt.plot(y_labels[0:len(scikitTimes)], scikitTimes, label = "Scikit")
plt.plot(y_labels[0:len(kubeTimes)], kubeTimes, label = "Kubernetes") 
plt.xlabel("Nº Samples")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig(f"{workdir}/plots/times-plot.png")


plt.figure()
plt.plot(y_labels[0:len(speedUps)], speedUps, label = "speedups")
plt.xlabel("Nº Samples")
plt.ylabel("SpeedUp")
plt.legend()
plt.savefig(f"{workdir}/plots/speedup-plot.png")