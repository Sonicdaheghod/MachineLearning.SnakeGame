#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Data of Students' Grades

# ### Idea based off of Dataquest
# ### https://youtu.be/Hr06nSA-qww

# In[1]:


#Bringing up initial data


# In[2]:


import pandas as pd


# In[3]:


game = pd.read_csv(r"C:\Users\Megan Tran\Desktop\Megan's USB\College\Code\Python\Machine Learning\snakes_count_1000.csv")
game


# In[26]:


#Data cleaning

game = game.dropna()
game


# In[27]:


#having model predict final grades of students based on provided data
from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[28]:


train = game[game["GameNumber"] < 900].copy()
#this determines how well model does
test = game[game["GameNumber"] >= 900].copy()



# In[29]:


train.shape


# In[30]:


test.shape

#90:10 split


# In[31]:


predictors = ["GameNumber"]
target = "GameLength"


# In[32]:


reg.fit(train[predictors], train[target])


# In[33]:


#Here the model predicts the length of the game for future games played based on the data provided


# In[34]:


predictions = reg.predict(test[predictors])
predictions


# In[47]:


#properly placing the predictions into the csv file


# In[48]:


test["Predictions Game Length"] = predictions
test


# In[49]:


#Round predicted times to two decimal points

test["Predictions Game Length"] = test["Predictions Game Length"].round(2)
test


# In[38]:


#Seeing if the predictions fit with the regression line


# In[16]:


import seaborn as sns

#we want this
sns.lmplot(x="GameNumber",y="GameLength",data=game, fit_reg=True,ci=None) 


# In[17]:


#weak negative correlation that prediction model is trying to follow


# ## Measuring error

# In[52]:


from sklearn.metrics import mean_absolute_error

my_error = mean_absolute_error(test["GameLength"], test["Predictions Game Length"])
my_error = my_error.round(2)
my_error


# In[ ]:


#this value shows that on average, the predictions were around 16.46 minutes from the actual time the game went on for   


# In[54]:


#to determine if the error is good or not, error < standard deviation

game.describe()["GameLength"]

#16.46 < 23.41, so error is good


# In[57]:


#to better understand our error value, let's see the error between the actual and predicted values for each game played

error_games = (test["GameLength"] - predictions).abs()
error_games = error_games.round(2)
error_games


# In[ ]:




