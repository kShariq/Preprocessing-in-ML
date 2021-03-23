#data preprocessing



#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 3].values

#missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])

#encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
columntransformer=ColumnTransformer([("Country",OneHotEncoder(),[0])],remainder='passthrough')
X=columntransformer.fit_transform(X)

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)










