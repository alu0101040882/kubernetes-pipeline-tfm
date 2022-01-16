
import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn import datasets

from kube_pipe_scikit import Kube_pipe, make_kube_pipeline

import os

import time
import datetime

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import csv

workdir = f"{os.path.dirname(os.path.realpath(__file__))}/test-results"


BASE_PROCS = 2

node_names = ["k3s-nodo-4-1cpu","k3s-nodo-5-1cpu","k3s-nodo-6-1cpu","k3s-nodo-7-1cpu","k3s-nodo-8-1cpu","k3s-nodo-9-1cpu","k3s-nodo-10-1cpu","k3s-nodo-11-1cpu","k3s-nodo-12-1cpu","k3s-nodo-13-1cpu"]


test_samples = [100,1000,2000,3000,4000,5000,10000,20000,30000,40000,50000,100000,200000,300000,400000,500000]
#test_samples = [10,10]

NUMBER_OF_FEATURES = 5

NUMBER_OF_TEST = 1


clasifiers = [  

                [StandardScaler(), LogisticRegression()],
                [StandardScaler(), DecisionTreeClassifier()],
                [StandardScaler(), RandomForestClassifier()],
                [StandardScaler(), AdaBoostClassifier()],
                [StandardScaler(), SVC(gamma=2, C=1)]
            
            ]


kubepipelines = make_kube_pipeline(*clasifiers)


scikitPipelines = []


clasifierNames = []

for clasifier in clasifiers:
    scikitPipelines.append(make_pipeline(*clasifier))
    clasifierNames.append(str(type(clasifier[-1]).__name__))


#Cordon all nodes
def cordonNodes(node_names):
    for node in node_names:
        os.system(f"kubectl cordon {node}")


def mean(arr):
    sum = 0

    for num in arr:
        sum+=num
    
    return sum/len(arr)

def test(pipelines,testTimes,X_train,y_train):
    times = []

    for i in range(testTimes):
        inicio = time.time()

        if(isinstance(pipelines,Kube_pipe)):
            pipelines.fit(X_train,y_train)
            pipelines.deleteTemporaryFiles()
        else:
            for pipeline in pipelines:
                pipeline.fit(X_train, y_train)

        fin = time.time()

        times.append(fin-inicio)

        str(datetime.timedelta(seconds=fin-inicio))
        print(times)

        return times

now = datetime.datetime.now().strftime("%d-%m_%H:%M")

os.mkdir(f"{workdir}/{now}")
os.mkdir(f"{workdir}/{now}/csv")
os.mkdir(f"{workdir}/{now}/plots")


kubenames = ["Samples","Scikit"]
speednames = ["Samples"]
for i, name in enumerate(node_names):
    kubenames.append(f"Kubernetes-{i+BASE_PROCS}proc")
    speednames.append(f"SpeedUp-{i+BASE_PROCS}proc")


with open(f"{workdir}/{now}/csv/times.csv", "a") as file:
    writer = csv.writer(file)
    writer.writerow(kubenames)

with open(f"{workdir}/{now}/csv/speedup.csv", "a") as file:
    writer = csv.writer(file)
    writer.writerow(speednames)

del kubenames
del speednames

scikitTimes = []
kubeTimes = []
speedUps = []

try:
    with open(f"{workdir}/{now}/summary.txt", "a") as f:
        f.write(f"Results of pipelines {clasifierNames}\n")

        for i , n_sample in enumerate(test_samples):
            X, y = datasets.make_classification(n_samples=n_sample,n_features=NUMBER_OF_FEATURES)

            f.write(f"{n_sample} samples:\n")

            cordonNodes(node_names)

            scikitTimes.append(mean(test(scikitPipelines,NUMBER_OF_TEST,X,y)))
            f.write(f"Scikit Pipeline:    \t {scikitTimes[i]} seconds\n")

            kubeTimes.append([])
            speedUps.append([])

            for proc, name in enumerate(node_names):

                os.system(f"kubectl uncordon {name}")

                kubeTimes[i].append(mean(test(kubepipelines,NUMBER_OF_TEST,X,y)))
                
                speedUps[i].append(scikitTimes[i]/kubeTimes[i][proc])
                

            f.write(f"Kubernetes Pipeline:\t {kubeTimes[i]} seconds\n")

            f.write(f"Speedup:            \t {speedUps[i]}\n")
                
            with open(f"{workdir}/{now}/csv/times.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow([n_sample,scikitTimes[i]]+kubeTimes[i])

            with open(f"{workdir}/{now}/csv/speedup.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow([n_sample]+speedUps[i])
        
            del X
            del y

            f.flush()
            os.fsync(f)

            print(f"samples:{n_sample}\nscikit: {scikitTimes[i]}\nkubernetes: {kubeTimes[i]}\nspeedup:{speedUps[i]}\n\n")

finally:

    import matplotlib.pyplot as plt
    import pandas as pd

    y_labels = []


    kube_proc_times = []

    speedups_proc = [] 

    for i in range(len(node_names)):
        kube_proc_times.append([])
        speedups_proc.append([])

    for i in range(len(kubeTimes)):

        for j in range(len(kubeTimes[i])):
            kube_proc_times[j].append(kubeTimes[i][j])
            speedups_proc[j].append(speedUps[i][j])


    for sample in test_samples:
        y_labels.append(str(sample))


    plt.figure()
    plt.plot(y_labels[0:len(scikitTimes)], scikitTimes, label = "Scikit")

    for i, times in enumerate(kube_proc_times):
        plt.plot(y_labels[0:len(times)], times, label = f"Kubernetes-{i+BASE_PROCS}procs")

    plt.xlabel("Nº Samples")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig(f"{workdir}/{now}/plots/times-plot.png")


    plt.figure()
    for i, speedup in enumerate(speedups_proc):
        plt.plot(y_labels[0:len(speedup)], speedup, label = f"speedup-{i+BASE_PROCS}procs")

    plt.xlabel("Nº Samples")
    plt.ylabel("SpeedUp")
    plt.legend()
    plt.savefig(f"{workdir}/{now}/plots/speedup-plot.png")

    