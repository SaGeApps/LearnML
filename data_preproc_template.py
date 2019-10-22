import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Import
dataset = pd.read_csv("dataCache\Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Missing Data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis= 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encode categorical variables
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:, 0])