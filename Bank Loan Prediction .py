#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import pandas as pd
import numpy as np


# In[16]:


train=pd.read_csv(r"C:\Users\91882\OneDrive\Desktop\Bank loan prediction_Stramlit\train.csv")
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})#label encoding done here #map funct of pandas is used
train.head()


# In[17]:


train.describe()


# ## Checking the missing values 

# In[ ]:





# In[9]:


train.isnull().sum()


# ## Preprocessing the data 

# In[19]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv(r"C:\Users\91882\OneDrive\Desktop\Bank loan prediction_Stramlit\test.csv")
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[21]:



data.shape


# In[22]:


data.describe()


# In[23]:


data.isnull().sum()


# In[24]:


#heatmap
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[25]:


#Label encoding for gender 
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[26]:


## Labelling 0 & 1 for Marrital status
data.Married=data.Married.map({'Yes':1,'No':0})


# In[27]:


data.Married.value_counts()


# In[28]:


## Labelling 0 & 1 for Dependents
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


# In[29]:


data.Dependents.value_counts()


# In[30]:


## Labelling 0 & 1 for Education Status
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


# In[31]:


data.Education.value_counts()


# In[32]:


## Labelling 0 & 1 for Employment status
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})


# In[33]:


data.Self_Employed.value_counts()


# In[34]:


data.Property_Area.value_counts()


# In[35]:


## Labelling 0 & 1 for Property area
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})


# In[36]:


data.Property_Area.value_counts()


# In[37]:


data.head()


# In[38]:


data.Credit_History.size


# In[39]:


#handle missing values
data.Credit_History.fillna(np.random.randint(0,2),inplace=True)


# In[40]:


data.isnull().sum()


# In[41]:


data.Married.fillna(np.random.randint(0,2),inplace=True)


# In[42]:


data.isnull().sum()


# In[43]:


## Filling with median
data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[44]:


## Filling with mean
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)


# In[45]:


data.isnull().sum()


# In[46]:


data.Gender.value_counts()


# In[47]:


## Filling Gender with random number between 0-2
from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)


# In[48]:


data.Gender.value_counts()


# In[49]:


## Filling Dependents with median
data.Dependents.fillna(data.Dependents.median(),inplace=True)


# In[50]:


data.isnull().sum()


# In[51]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)


# In[52]:


data.isnull().sum()


# In[53]:


data.head()


# In[54]:


## Dropping Loan ID from data, it's not useful
data.drop('Loan_ID',inplace=True,axis=1)


# In[55]:


data.isnull().sum()


# In[56]:


data.head()


# ## Train Test Split

# In[57]:


train_X=data.iloc[:614,] ## all the data in X (Train set)
train_y=Loan_status  ## Loan status will be our Y


# In[58]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=0)


# In[59]:


train_X.head()


# In[60]:


test_X.head()


# ## Importing ML model from Sklearn

# In[61]:



from sklearn.linear_model import LogisticRegression


# ## Fit the model

# In[62]:


models=[]
models.append(("Logistic Regression",LogisticRegression()))


# In[63]:


scoring='accuracy'


# In[64]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


# In[65]:


for name,model in models:
    kfold=KFold(n_splits=10,random_state=0)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print(model)
    print("%s %f" % (name,cv_result.mean()))


# In[66]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

LR=LogisticRegression()
LR.fit(train_X,train_y)
pred=LR.predict(test_X)
print("Model Accuracy:- ",accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[67]:


print(pred)


# In[68]:


X_test=data.iloc[614:,]


# In[69]:


X_test.head()


# In[70]:


prediction = LR.predict(X_test)


# In[71]:


print(prediction)


# In[72]:


### data taken from the dataset
t = LR.predict([[0.0, 0.0, 0.0,	1, 0.0, 1811, 1666.0, 54.0, 360.0, 1.0, 2]])


# In[73]:


print(t)


# In[86]:


import pickle
from sklearn import svm
svc = svm.SVC()
# now you can save it to a file
file = "C:\\Users\91882\OneDrive\Desktop\Bank loan prediction_Stramlit\Model\model.pkl"
with open(file,'wb') as f:

    pickle.dump(svc, f)


# In[87]:


with open(file, 'rb') as f:
    k = pickle.load(f)


# In[ ]:




