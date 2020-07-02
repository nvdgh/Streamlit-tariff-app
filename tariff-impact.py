#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

from cforest.forest import CausalForest
from sklearn.model_selection import train_test_split

@st.cache
def load_image():
    return Image.open("bike.png")

@st.cache
def init_cf():
    cfparams = {
        'num_trees': 40,
        'split_ratio': 1,
        'num_workers': 4,
        'min_leaf': 5,
        'max_depth': 20,
        'seed_counter': 1,
        'use_transformed_outcomes': False
    }

    cf = CausalForest(**cfparams)
    cf.load("fitted_model.csv")

    return cf

@st.cache
def load_data():
    return pd.read_csv("cross_import_cn.csv")

@st.cache
def get_train_test(data):
    return train_test_split(X, y, t, test_size=0.2)

@st.cache
def get_errors(cf, x, y):
    return np.sqrt(mean_squared_error(y, cf.predict(x)))

#import the data
data = load_data()
st.title("Welcome to the Prediction App of Tariff Impact on Changes in Import Value Percentage")
st.image(load_image(), use_column_width=True)


#checking the data
st.write("This is an application for knowing how much the range of import values (quantity imported time product price) change due to 2018 tariff using Causal Forest. Let's try and see!")
check_data = st.checkbox("See the simple data")
if check_data:
    st.write(data.head())
st.write("Now let's find out how much the import values when we choosing some parameters.")

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

# create the input array
newx = np.array([[va_y, va_l, pl_l, inter_y, sk_l, m_l, k_l, rental_l,temp_l, it_l, mkt_l, outsource_l, tax_l, cn_mnc_ratio]])


X = data[['va_y', 'va_l', 'pl_l', 'inter_y', 'sk_l',
       'm_l', 'k_l', 'rental_l', 'temp_l', 'it_l', 'mkt_l', 'outsource_l',
       'tax_l', 'cn_mnc_ratio']]
y = data['crossiv']
t = data['Treated'].astype('bool')

x_train, x_test, y_train, y_test, treat_train, treat_test = get_train_test(data)

cf = init_cf()
errors = get_errors(cf, x_test, y_test)
predictions = cf.predict(newx)

#checking prediction house price
if st.button("Run me!"):
    st.header("Your business's import values will change an amount of {}%".format(np.round(predictions*10,2)))
    st.subheader("Your range of prediction is {}% - {}%".format(np.round(predictions*10-errors,2),np.round(predictions*10+errors,2)))
