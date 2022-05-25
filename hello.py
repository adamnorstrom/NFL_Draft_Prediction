#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import warnings
warnings.simplefilter("ignore")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets


# In[2]:


draft=pd.read_csv('draft_num_2016.csv')


# In[3]:


draft.head(60)


# In[4]:


draft.shape


# In[5]:


rating=pd.read_csv("Career_P_rating.csv")


# In[6]:


rating.head(251)


# In[7]:


rating['Player'] = rating['Player'].str.replace(r'\*', '')


# In[8]:


nr=rating.merge(draft, on='Player')


# In[9]:


nr.shape


# In[12]:


td=pd.read_csv('td_all.csv')


# In[13]:


td.head()


# In[14]:


td['Player'] = td['Player'].str.replace(r'\*', '')


# In[15]:


w_td_nr3=nr.merge(td, on='Player')


# In[16]:


w_td_nr3.head(60)


# In[17]:


w_td_nr3.shape


# In[18]:


w_td_nr3=w_td_nr3.drop(columns=['Rk'])


# In[19]:


w_td_nr3.head(77)


# In[20]:


print(w_td_nr3)


# In[34]:


nw_td_nr3=w_td_nr3.drop(['Player','Rnd','Year'],axis=1)


# In[35]:


nw_td_nr3.head(9)


# In[36]:


model1 = lm.LinearRegression()
X = nw_td_nr3.drop(columns=['Pick'])
# our y, the pick number
y = nw_td_nr3['Pick']
# fit model based on data
model1.fit(X, y)


# In[37]:


def rmse(target, pred):
    return np.sqrt(mean_squared_error(target, pred)) 


# In[38]:


predictions = model1.predict(X)
actual = y
print(rmse(actual, predictions))
print(np.mean(actual), np.std(actual))


# In[39]:


plt.hist(y)
plt.xlabel("pick")
plt.show()


# In[40]:


non_ex_set= nw_td_nr3[nw_td_nr3['Pick']< 150]

model2=lm.LinearRegression()

x= non_ex_set.drop(columns=['Pick'])
yp= non_ex_set['Pick']
model2.fit(x,yp)
prediction=model2.predict(x)
actuals=yp
print(rmse(actuals,prediction))
print(np.mean(actuals),np.std(actuals))


# In[50]:


X,Y = np.meshgrid(np.arange(0,170,1), np.arange(0,200,1))
Z = -1.0889273613441992* X + -0.033260975581398386 * Y + 215.78500938953354

ax = plt.figure(figsize=(8,8)).add_subplot(111, projection='3d')
ax.scatter(non_ex_set["TD"], 
           non_ex_set["Rate"], 
           non_ex_set['Pick'])
ax.plot_surface(X, Y, Z, alpha=0.5)
plt.xlabel("TD")
plt.ylabel("Rate")
ax.set_zlabel("Pick")
plt.title("Draft Predictions", pad=30, size=15);


# In[41]:


residual = actuals-prediction
plt.hist(residual)
plt.xlabel("residual(draft pick)")
plt.show()
print(np.mean(residual))


# In[ ]:


Overall, the model has a rmse of 41 which means that the predicted pick number is still on average a full round of the draft away from the actual pick number.  


# In[45]:


print([model2.intercept_, model2.coef_[0], model2.coef_[1]])


# In[ ]:




