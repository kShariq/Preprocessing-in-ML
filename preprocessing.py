#data preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 3].values


#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""














