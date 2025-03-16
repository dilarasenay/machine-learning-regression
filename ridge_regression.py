#!/usr/bin/env python
# coding: utf-8

# ### RIDGE REGRESSION

# In[46]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\dilara\Downloads\miuul makine ogrenmesi\datasets\hitters.csv")
data=df.copy()
data=data.dropna()
ms=pd.get_dummies(df[['League', 'Division','NewLeague']])
y=df["Salary"]
X_=df.drop(['League', 'Division','NewLeague'],axis=1).astype("float")
X = pd.concat([X_, ms], axis=1)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=42)
X_train = X_train.dropna()
y_train = y_train.dropna()
X_test = X_test.dropna()
y_test = y_test.dropna()


# In[47]:


from sklearn.linear_model import Ridge 


# In[48]:


ridge_model = Ridge(alpha=0.1).fit(X_train, y_train)
ridge_model


# In[49]:


ridge_model.coef_


# In[50]:


10**np.linspace(10,-2,100)*0.5


# In[51]:


lambdalar=10**np.linspace(10,-2,100)*0.5


# In[52]:


ridge_model=Ridge()
coefficients = []

for i in lambdalar : 
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train,y_train)
    coefficients.append(ridge_model.coef_)
    
ax = plt.gca()
ax.plot(lambdalar,coefficients)
ax.set_xscale('log')

plt.xlabel("Lambda(Alpha) Values")
plt.ylabel("Coefficients/Weights")
plt.title("Ridge Coefficients as a Function of Regularization")


# ### TAHMİN

# In[53]:


y_pred = ridge_model.predict(X_test)


# In[54]:


np.sqrt(mean_squared_error(y_test , y_pred))


# ### MODEL TUNİNG 

# In[55]:


lambdalar=10**np.linspace(10,-2,100)*0.5


# In[56]:


lambdalar[0:5]


# In[61]:


from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


ridge_cv = RidgeCV(alphas = lambdalar , 
                  scoring = "neg_mean_squared_error")
                  #normalize = True)


# In[62]:


ridge_cv.fit(X_train , y_train)


# In[63]:


ridge_cv.alpha_


# In[70]:


ridge_tuned = Ridge(alpha=ridge_cv.alpha_).fit(X_train , y_train)
                   #normalize=True)


# In[71]:


np.sqrt(mean_squared_error(y_test ,ridge_tuned.predict(X_test) ))


# In[ ]:




