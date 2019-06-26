import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style 
from sklearn.preprocessing import OneHotEncoder
import lightgbm

data = pd.read_csv("renfe.csv")
#print(data.head())

for col in ['insert_date','start_date','end_date']:
    date_col = pd.to_datetime(data[col])
    data[col + '_hour'] = date_col.dt.hour
    data[col + '_minute'] = date_col.dt.minute
    data[col + '_second'] = date_col.dt.second
    data[col + '_weekday'] = date_col.dt.weekday
    data[col + '_day'] = date_col.dt.day
    data[col + '_month'] = date_col.dt.month
    data[col + '_year'] = date_col.dt.year
    del data[col]
#print(data.head())    

data.isnull().sum()
data.dropna(inplace=True)

for col in data.columns:
    print(col,":",data[col].unique().shape[0])
    
data_drop = [col  for col in data.columns if data[col].unique().shape[0]==1]
data.drop(columns=data_drop,inplace=True)

x = 0
for col in data.columns:
    if x==0:
        data_dr = [col]
        x=1
data.drop(columns = data_dr , inplace = True)        


cor = data.corr(method='pearson')
cor.style.background_gradient(cmap='coolwarm')
#The only highly correlated feature we can observe is the between the start and end date (weekday, day and month). We can drop off one of each.

data.drop(columns = ['end_date_weekday','end_date_day','end_date_month'],inplace=True)

X = data.drop(columns=['price'])
y = data['price']
encoder = OneHotEncoder()
X = encoder.fit_transform(X.values)
best = 0

####here i use linear regression to predict the price but it seems like it's not very accurate

linear = sklearn.linear_model.LinearRegression()
# for w in range(5):
#     print(w)
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)
linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)
print(acc)
#     if acc>best:
#         best=acc
#         print("best:",best)            

####here i use SVM's LinearSVR to predict the price but it seems like it's not very accurate

linear = sklearn.svm.LinearSVR()
# for w in range(5):
#     print(w)
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)
linear.fit(x_train, y_train)
acc = linear.score(x_test,y_test)
print(acc)
#     if acc>best:
#         best=acc
#         print("best:",best)            

####here i use We will now try to use a gradient boosting machine, 
##  which is a method that combines a collection of trees (i.e. an ensemble method) to make a well-supported prediction. 

gbm = lightgbm.LGBMRegressor()
gbm.fit(x_train , y_train)
print("Train Score:",gbm.score(x_train , y_train))
print("Test Score:",gbm.score(x_test , y_test))
# acc = gbm.score(x_test , y_test)
# print(acc)
