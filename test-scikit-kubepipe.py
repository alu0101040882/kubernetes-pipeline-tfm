


from os import pipe
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import datasets

from kube_pipe_scikit import make_kube_pipeline


import time

iris = datasets.load_iris()

X_train,X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2)

NUMBER_OF_TEST = 1

times = []

for i in range(NUMBER_OF_TEST):
    inicio = time.time()

    pipeline = make_kube_pipeline([OneHotEncoder(handle_unknown="ignore"), LogisticRegression()],
                                [OneHotEncoder(handle_unknown="ignore"), RandomForestClassifier()])


    model = pipeline.fit(X_train,y_train)

    fin = time.time()

    times.append(fin-inicio)


print(times)


