# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('data.csv')
dataset=dataset.drop(['Unnamed: 32'],axis=1)
sns.heatmap(dataset.corr())
Co=dataset.corr()

#Encoding
from sklearn.preprocessing import LabelEncoder
le_X=LabelEncoder()
for i in list(dataset.columns):
    if dataset[i].dtype=='object':
        dataset[i]=le_X.fit_transform(dataset[i])
        
#Getting DV AND IV
y=dataset['diagnosis']
x=dataset.drop('diagnosis',axis=1)

#Train Test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
list_1=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_s=knn.predict(x_test)
    scores=accuracy_score(y_test,pred_s)
    list_1.append(scores)
    
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)
lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
score_1=accuracy_score(y_test,pred_1)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
pred_2=rfc.predict(x_test)
score_2=accuracy_score(y_test,pred_2)

