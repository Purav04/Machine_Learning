# kaggle notebook link:https://www.kaggle.com/purav04/notebook8f7c1510c6 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# train and testing dataset
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.isnull().sum()
test_data.isnull().sum()

X1 = train_data.drop(columns = ['Name','PassengerId','Cabin','Survived'],axis=1)
y1 = train_data.Survived
X_test = test_data.drop(columns =  ['PassengerId','Name','Ticket','Cabin'],axis=1)

sns.heatmap(X1.corr(),annot=True)

# split dataset into train and valid dataset

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X1,y1,train_size=0.8,test_size=0.2,random_state=1)

# deal with null values

X_train.isnull().sum()
X_test.isnull().sum()

from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

for col in ['Age','Fare']:
    X_train[col] = num_imputer.fit_transform(X_train[[col]])
    X_valid[col] = num_imputer.fit_transform(X_valid[[col]])
    X_test[col] = num_imputer.fit_transform(X_test[[col]])
X_train['Embarked'] = cat_imputer.fit_transform(X_train[['Embarked']].values)

X_train = X_train.drop(columns = ['Ticket'],axis=1)
X_valid = X_valid.drop(columns = ['Ticket'],axis=1)
X_train.head()

# feature engineering
def family(df):
    df['Family'] = df['SibSp'] + df['Parch']
    df = df.drop(columns=['SibSp','Parch'],axis=1)
    return df
X_train = family(X_train)
X_valid = family(X_valid)
X_test = family(X_test)

# one hot encoding
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown="ignore",sparse=False)

def one_hot(X):
    ohe_cols = ['Pclass','Sex','Embarked']
    ohe_cols_train = pd.DataFrame(one_hot_encoder.fit_transform(X[ohe_cols]))
    ohe_cols_train.index = X.index
    XX = X.drop(ohe_cols,axis=1)
    return pd.concat([XX,ohe_cols_train],axis=1)
X_ohe_train = one_hot(X_train)
X_ohe_valid = one_hot(X_valid)
X_ohe_test = one_hot(X_test)

X_ohe_train.rename(columns={0:"Pclass_1",1:"Pclass_2",2:"Pclass_3",3:"Sex_female",4:"Sex_male",5:"Embarked_C",6:"Embarked_Q",7:"Embarked_S"},inplace=True)
X_ohe_valid.rename(columns={0:"Pclass_1",1:"Pclass_2",2:"Pclass_3",3:"Sex_female",4:"Sex_male",5:"Embarked_C",6:"Embarked_Q",7:"Embarked_S"},inplace=True)
X_ohe_test.rename(columns={0:"Pclass_1",1:"Pclass_2",2:"Pclass_3",3:"Sex_female",4:"Sex_male",5:"Embarked_C",6:"Embarked_Q",7:"Embarked_S"},inplace=True)

sns.heatmap(X_ohe_test.corr(),annot=True)

# normalization 
from sklearn.preprocessing import MinMaxScaler
normalize = MinMaxScaler()

def normalize_feature(df):
    normalize_cols = ['Age','Fare']
    df_new = pd.DataFrame()
    for cols in normalize_cols:
        X_scaled = normalize.fit_transform(np.array(df[cols]).reshape(-1,1))
        df_scaled = pd.DataFrame(X_scaled,columns = [cols+'_new'])
        df_new = pd.concat([df_scaled,df_new],axis=1)
    df = pd.concat([df,df_new],axis=1)
    df.drop(columns=['Age','Fare'],axis=1,inplace=True)
    return df
    
X_new_train = normalize_feature(X_ohe_train)
X_new_valid = normalize_feature(X_ohe_valid)
X_new_test  = normalize_feature(X_ohe_test)

sns.heatmap(X_new_train.corr(),annot=True)

# build models 

# random forest classifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
model = RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=5,random_state=1,oob_score=True)
model.fit(X_ohe_train,y_train)
model.score(X_ohe_valid,y_valid)
pred = model.predict(X_ohe_valid)
print(mean_absolute_error(pred,y_valid))

# logistic regression
from sklearn.linear_model import LogisticRegression
lsc_r = LogisticRegression(penalty='l2',random_state=1,solver='lbfgs',tol=0.001).fit(X_ohe_train,y_train)
lsc_r.score(X_ohe_valid,y_valid)

# support vector machine
from sklearn import svm
clf_svm = svm.NuSVC().fit(X_ohe_train,y_train)
clf_svm.score(X_ohe_valid,y_valid)
