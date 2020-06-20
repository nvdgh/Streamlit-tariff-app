#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

from cforest.forest import CausalForest
from sklearn.model_selection import train_test_split


# In[4]:


#import the data
data = pd.read_csv('d:/git/InSightProject/data/processed/cross_import_cn.csv')
image = Image.open("d:/git/InSightProject/streamlit_folder/bike.png")
st.title("Welcome to the Prediction App of Tariff Impact on Changes in Import Value Percentage")
st.image(image, use_column_width=True)


# In[5]:


#checking the data
st.write("\n This help you to quantify how much is the range of import values (quantity imported time product price) change due to 2018 tariff using Causal Forest. Let's try and see! \n")
check_data = st.checkbox("\n See the simple data \n")
if check_data:
    st.write(data.head())
st.write("\n Now let's find out how much the import values when we choosing some parameters. \n")


# In[6]:


#input the numbers
va_y = data.va_y.mean()
va_l = data.va_l.mean()
pl_l = st.slider("What is your business's proportion of production workers?",data.pl_l.min(), data.pl_l.max(),data.pl_l.mean())
inter_y = data.inter_y.mean()
sk_l     = st.slider("What is your business's skill intensity?",data.sk_l.min(), data.sk_l.max(),data.sk_l.mean())
m_l = st.slider("What is your business's skill intensity?",data.m_l.min(), data.m_l.max(),data.m_l.mean())
k_l = data.k_l.mean()
rental_l = data.rental_l.mean()
temp_l      = st.slider("What is your business's temporary workers intensity?",data.temp_l.min(), data.temp_l.max(),data.temp_l.mean())
it_l = data.it_l.mean()
mkt_l = data.mkt_l.mean()
outsource_l    = st.slider("What is your business's outsourcing intensity?",data.outsource_l.min(), data.outsource_l.max(),data.outsource_l.mean())
tax_l = data.tax_l.mean()
cn_mnc_ratio    = st.slider("What is your business's multinational corporation ratio in China?",data.cn_mnc_ratio.min(), data.cn_mnc_ratio.max(),data.cn_mnc_ratio.mean())


# In[7]:


# create the input array
newx = np.array([[va_y, va_l, pl_l, inter_y, sk_l, m_l, k_l, rental_l,temp_l, it_l, mkt_l, outsource_l, tax_l, cn_mnc_ratio]])


# In[8]:


X = data[['va_y', 'va_l', 'pl_l', 'inter_y', 'sk_l',
       'm_l', 'k_l', 'rental_l', 'temp_l', 'it_l', 'mkt_l', 'outsource_l',
       'tax_l', 'cn_mnc_ratio']]
y = data['crossiv']
t = data['Treated'].astype('bool')


# In[9]:


x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(X, y, t, test_size=0.2)


# In[10]:


cfparams = {
    'num_trees': 40,
    'split_ratio': 1,
    'num_workers': 4,
    'min_leaf': 5,
    'max_depth': 20,
    'seed_counter': 1,
}


# In[11]:


cf = CausalForest(**cfparams)


# In[12]:


# cf = cf.fit(x_train, treat_train, y_train)


# In[14]:


# cf.save('d:/git/InSightProject/data/processed/cf-tariff-saved.csv')


# In[15]:


cff = CausalForest(**cfparams)
cff = cff.load("d:/git/InSightProject/data/processed/cf-tariff-saved.csv")


# In[16]:


errors = np.sqrt(mean_squared_error(y_test,cff.predict(x_test)))


# In[17]:


predictions = cff.predict(newx)


# In[18]:


#checking prediction house price
if st.button("Run me!"):
    st.header("Your business's import values will change an amount of {}%".format(np.round(predictions*10,2)))
    st.subheader("Your range of prediction is {}% - {}%".format(np.round(predictions*10-errors,2),np.round(predictions*10+errors,2)))

