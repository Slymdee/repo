#!/usr/bin/env python
# coding: utf-8

# ## Online Payments Fraud Detection Dataset Case Study
# 
# 
# As we are approaching modernity, the trend of paying online is increasing tremendously. It is very beneficial for the buyer to pay online as it saves time, and solves the problem of free money. Also, we do not need to carry cash with us. But we all know that Good thing are accompanied by bad things. 
# 
# The online payment method leads to fraud that can happen using any payment app. That is why Online Payment Fraud Detection is very important.

# In[48]:


#Importing Libraries and Datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# The below column reference:
# • step: represents a unit of time where 1 step equals 1
# hour
# • type: type of online transaction
# • amount: the amount of the transaction
# • nameOrig: customer starting the transaction
# • oldbalanceOrg: balance before the transaction
# • newbalanceOrig: balance after the transaction
# • nameDest: recipient of the transaction
# • oldbalanceDest: initial balance of recipient
# before the transaction
# • newbalanceDest: the new balance of the
# recipient after the transaction
# • isFraud: fraud transaction
# 

# In[2]:


data=pd.read_csv(r'C:\Users\Admin\Downloads\Online Payment Fraud Detection.csv')
data.head()


# In[3]:


print(data.isnull().sum())


# In[4]:


# Exploring transaction type
print(data.type.value_counts())


# In[5]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()


# In[43]:


# Let’s see the count plot of the Payment type column using Seaborn library.

sns.countplot(x='type', data=data)


# In[44]:


# bBar plot for analyzing Type and amount column simultaneously.
sns.barplot(x='type', y='amount', data=data)


# In[45]:


# let’s see the distribution of the step column using distplot.

plt.figure(figsize=(15, 6))
sns.distplot(data['step'], bins=50)


# In[46]:


#Let’s find the correlation among different features using Heatmap.

plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)


# In[7]:


# Checking correlation
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))


# In[8]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


# In[9]:


# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[10]:


# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[12]:


# training a machine learning model
from sklearn.linear_model import LogisticRegression
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = LogisticRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[37]:


# training a machine learning model
from sklearn.ensemble import RandomForestClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[42]:


from sklearn.metrics import plot_confusion_matrix
 
plot_confusion_matrix(model, xtest, ytest)
plt.show()


# Summary

# So this is how we can detect online payments fraud with machine learning using Python. Detecting online payment frauds is one of the applications of data science in finance.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




