import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder




from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import datasets

import time

iris = datasets.load_iris()

X_train,X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2)


NUMBER_OF_TEST = 1

times = []


for i in range(NUMBER_OF_TEST):
    inicio = time.time()

    #Creación de los pipelines
    LogisticRegressionPipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'),LogisticRegression())
    RandomForestPipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'),RandomForestClassifier())

    #Creacion de la lista de los pipelines
    pipelines = [LogisticRegressionPipeline,RandomForestPipeline] 

    for pipeline in pipelines:
        pipeline.fit(X_train, y_train)

    fin = time.time()

    times.append(fin-inicio)


print(times)


