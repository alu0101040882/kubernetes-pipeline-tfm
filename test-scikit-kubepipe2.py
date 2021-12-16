


from os import pipe
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets

from kube_pipe_scikit import Kube_pipe, make_kube_pipeline
from sklearn.tree import DecisionTreeClassifier
import os

import time
import datetime


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

datasets.fetch_california_housing

workdir = f"{os.path.dirname(os.path.realpath(__file__))}/test-results"

test_samples = [100, 1000]


NUMBER_OF_FEATURES = 10

NUMBER_OF_TEST = 1


#Kubernetes pipeline
kubepipelines = make_kube_pipeline(
                                    [StandardScaler(), LogisticRegression()],
                                    [StandardScaler(), DecisionTreeClassifier()],
                                    [StandardScaler(), RandomForestClassifier()],
                                    [StandardScaler(), GaussianProcessClassifier(1.0 * RBF(1.0))],
                                    [StandardScaler(), SVC(gamma=2, C=1)],
                                    [StandardScaler(), SVC(kernel="linear", C=0.025)],
                                    [StandardScaler(), AdaBoostClassifier()],
                                    [StandardScaler(), GaussianNB()],
                                    [StandardScaler(), QuadraticDiscriminantAnalysis()]
                                    
                                )

#Scikit pipeline
scikitPipelines =   [   
                        make_pipeline(StandardScaler(), LogisticRegression()),
                        make_pipeline(StandardScaler(), DecisionTreeClassifier()),
                        make_pipeline(StandardScaler(), RandomForestClassifier()),
                        make_pipeline(StandardScaler(), GaussianProcessClassifier(1.0 * RBF(1.0))),
                        make_pipeline(StandardScaler(), SVC(gamma=2, C=1)),
                        make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025)),
                        make_pipeline(StandardScaler(), AdaBoostClassifier()),
                        make_pipeline(StandardScaler(), GaussianNB()),
                        make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
                        
                    ]

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
        else:
            for pipeline in pipelines:
                pipeline.fit(X_train, y_train)

        fin = time.time()

        times.append(fin-inicio)

        str(datetime.timedelta(seconds=fin-inicio))
        print(times)

        return times


randomX, randomy = datasets.make_classification(n_samples=1000000,n_features=10)
now = datetime.datetime.now().strftime("%d-%m_%H:%M")

os.mkdir(f"{workdir}/{now}")
os.mkdir(f"{workdir}/{now}/csv")
os.mkdir(f"{workdir}/{now}/plots")


with open(f"{workdir}/{now}/csv/times.csv", "a") as file:
    writer = csv.writer(file)
    writer.writerow(["Scikit","Kubernetes"])

with open(f"{workdir}/{now}/csv/speedup.csv", "a") as file:
    writer = csv.writer(file)
    writer.writerow(["Speed Up"])


try:
    with open(f"{workdir}/{now}/resumen.txt", "a") as f:
        f.write(f"Results of pipelines \n")

        for n_sample in test_samples:
            X, y = datasets.make_classification(n_samples=n_sample,n_features=NUMBER_OF_FEATURES)
            f.write(f"{n_sample} samples:\n")

            scikitTime = mean(test(scikitPipelines,NUMBER_OF_TEST,X,y))
            f.write(f"Scikit Pipeline:    \t {str(datetime.timedelta(seconds=scikitTime))}  ({scikitTime} seconds)\n")

            kubeTime = mean(test(kubepipelines,NUMBER_OF_TEST,X,y))
            f.write(f"Kubernetes Pipeline:\t {str(datetime.timedelta(seconds=kubeTime))}  ({kubeTime} seconds)\n")

            speedUp = scikitTime/kubeTime
            f.write(f"Speedup:            \t {speedUp}\n\n")

            print(f"samples:{n_sample}\nscikit: {scikitTime}\nkubernetes: {kubeTime}\nspeedup:{speedUp}\n\n")

            with open(f"{workdir}/{now}/csv/times.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow([scikitTime,kubeTime])

            with open(f"{workdir}/{now}/csv/speedup.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow([speedUp])
        
            f.flush()
            os.fsync(f)

finally:

    import matplotlib.pyplot as plt
    import pandas as pd

    speedup = pd.read_csv(f"{workdir}/{now}/csv/speedup.csv")

    times = pd.read_csv(f"{workdir}/{now}/csv/times.csv")

    speedup.plot()
    plt.savefig(f"{workdir}/{now}/plots/speedup-plot.png")

    times.plot()
    plt.savefig(f"{workdir}/{now}/plots/times-plot.png")