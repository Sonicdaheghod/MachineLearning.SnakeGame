#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Data of Students' Grades

# In[1]:


#Bringing up initial data


# In[2]:


import pandas as pd


# In[16]:


game = pd.read_csv(r"C:\Users\Megan Tran\Desktop\Megan's USB\College\Code\Python\Machine Learning\snakes_count_1000.csv")
game


# In[17]:


#Data cleaning

game = game.dropna()
game


# In[18]:


#having model predict final grades of students based on provided data
from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[19]:


train = game[game["GameNumber"] < 500].copy()
#this determines how well model does
test = game[game["GameNumber"] >= 500].copy()



# In[20]:


train.shape


# In[8]:


test.shape


# In[21]:


predictors = ["GameNumber"]
target = "GameLength"


# In[22]:


reg.fit(train[predictors], train[target])


# In[ ]:


#Here the model predicts the length of the game for future games played based on the data provided


# In[23]:


predictions = reg.predict(test[predictors])
predictions


# In[ ]:





# In[ ]:




