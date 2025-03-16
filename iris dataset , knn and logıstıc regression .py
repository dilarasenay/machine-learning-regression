#!/usr/bin/env python
# coding: utf-8

# ### KNN WITH IRIS DATASET

# In[1]:


from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics  import confusion_matrix, classification_report, precision_score, recall_score


# In[2]:


irisdata = load_iris()
irisdata


# In[3]:


irisdata.data


# In[4]:


irisdata.target


# In[5]:


print(irisdata.DESCR)


# In[6]:


irisdata.target_names


# In[7]:


irisdata.feature_names


# In[8]:


X=irisdata.data
y=irisdata.target


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.35, random_state=12)


# In[10]:


neighbors = np.arange(1, 9)
neighbors


# In[11]:


test_accuracy = np.zeros(len(neighbors))
print(test_accuracy)


# In[12]:


for i , k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    
    test_accuracy[i]=knn.score(X_test , y_test)

plt.plot(neighbors , test_accuracy , label="Testing dataset Accuracy")
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.show()


# In[13]:


y_pred_knn = knn.predict(X_test)
confusion_matrix(y_test , y_pred_knn)


# In[15]:


report = classification_report(y_test, y_pred_knn, target_names=irisdata.target_names)
print(report)


# ### LOGISTIC REGRESSION WITH IRIS DATASET

# In[16]:


log_reg=LogisticRegression(max_iter=1000)  
log_reg.fit(X_train, y_train)
y_pred=log_reg.predict(X_test) 


# In[17]:


log_reg_score=log_reg.score(X_test, y_test)


# In[19]:


log_reg_score


# In[ ]:




