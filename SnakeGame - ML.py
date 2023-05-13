#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Credit to https://youtu.be/Hr06nSA-qww?t=1022


# In[2]:


import pandas as pd


# # Access Data

# ### Accessed data issue resolved by adding "r" before quotes

# In[3]:


teams = pd.read_csv(r"C:\Users\Megan Tran\Desktop\Megan's USB\College\Code\Python\Machine Learning\teams.csv")
teams


# ## Keeping Certain Columns

# In[4]:


teams = teams[["team", "country", "year", "athletes","age","prev_medals", "medals"]]
teams


# ## Finding correlation
# 
# #### The greater the r value, the better a prediction can be made for said variable 

# In[5]:


teams.corr()["medals"]


# ## Plotting data

# In[6]:


#a - correlation of athletes x medals


# In[7]:


import seaborn as sns
sns.lmplot(x="athletes",y="medals",data=teams, fit_reg=True,) 

#we want this
sns.lmplot(x="athletes",y="medals",data=teams, fit_reg=True,ci=None) 


# In[8]:


#b - correlation of athletes x age


# In[9]:


sns.lmplot(x="athletes",y="age",data=teams, fit_reg=True,ci=None) 


# ## Histogram of data athlete x medals

# In[10]:


teams.plot.hist(y="medals")


# In[11]:


#Here, the graph is not evenly balanced because there are a lot of medals earned in the 0-100 range.


# ## Data Cleaning

# In[12]:


teams[teams.isnull().any(axis=1)]


# In[13]:


#remove rows that don't have data on medals


# In[14]:


teams = teams.dropna()
teams


# In[15]:


#respect order of data for training since future data cannot be used for prediction


# In[16]:


train = teams[teams["year"] < 2012].copy()
#this determines how well model does
test = teams[teams["year"] >= 2012].copy()





# In[17]:


train.shape


# In[18]:


test.shape


# In[19]:


#above is an 80 / 20 split which is normal


# ## Training our Model

# In[20]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[21]:


#training regression model to predict medals using predictors from data set
predictors = ["athletes", "prev_medals"]
target = "medals"


# In[22]:


reg.fit(train[predictors], train[target])


# In[23]:


#using above model to make predictions, cannot pass actual values bc that would give the model the answers which does not train it
predictions = reg.predict(test[predictors])


# In[24]:


predictions


# In[25]:


#from the predictions of the medals, the model outputs negative and decimal numbers.
#this makes no sense bc you cannot win a part of a medal nor win a negative number of medals

#fixing the results


# In[26]:


test["predictions"] = predictions
test


# In[27]:


#fix that if the predictions for medals is less than zero, that value will just be zero on display

test.loc[test["predictions"] < 0, "predictions"] = 0




# In[28]:


test["predictions"] = test["predictions"].round()


# In[29]:


test


# In[ ]:




