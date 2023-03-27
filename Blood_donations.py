#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
transfusion=pd.read_csv("transfusion.data")
transfusion.head()


# In[8]:


transfusion.info()


# In[10]:


transfusion.rename(columns={'whether he/she donated blood in March 2007': 'target'},inplace=True)
transfusion.head(2)


# In[11]:


transfusion.target.value_counts(normalize=True).round(3)


# In[14]:


from sklearn.model_selection import train_test_split
X=transfusion.drop("target",axis=1)
y=transfusion["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
X_train.head(2)


# In[19]:


from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

tpot=TPOTClassifier(generations=5,population_size=20,verbosity=2,scoring='roc_auc',random_state=42,disable_update_check=True,config_dict='TPOT light')
tpot.fit(X_train,y_train)
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')


# In[20]:


print(X_train.var().round(3))


# In[21]:


import numpy as np
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()
col_to_normalize = 'Monetary (c.c. blood)'
for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)
    print(X_train_normed.var().round(3))


# In[22]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)
logreg.fit(X_train_normed, y_train)
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# In[23]:


from operator import itemgetter
sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    reverse=True
)


# In[ ]:




