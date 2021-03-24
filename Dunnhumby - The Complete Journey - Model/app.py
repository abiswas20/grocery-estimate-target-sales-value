import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from joblib import dump, load
import streamlit as st


def func_man_id(m_p):
    all_plants={}
    all_plants={'531':0, '709':0, '865':0, '926':0, '927':0, '1078':0, '1453':0, '1487':0, '1543':0, '5423':0}
    if m_p in all_plants:
        all_plants[m_p]=1
        print(all_plants)
        return all_plants
    else:
        print('manufacturer not in list. please re-check manufacturer id and re-enter')


#m_p=st.text_input("Input the Manufacturer ID \n1543, 926, 1078, 927, 5423, 1487, 1453, 531, 865, 709", 1)
#'RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC', 'DAY', 'VOLUME','Private', 'FLUID MILK WHITE ONLY', 'MISCELLANEOUS MILK'
#m_p=input('manufacturer id:')
#func_man_id(m_p)

##### Change this to match input #####
retail_disc = float(st.text_input("retail discount",0.05))
coupon_disc = float(st.text_input("coupon discount",0.01))
coupon_match_disc = float(st.text_input("coupon match discount",0))
day = int(st.text_input("day", 0))
volume = int(st.text_input("volume",128))
brand = st.selectbox("private brand?",(0,1))
fluid_white_milk=st.selectbox("FLUID MILK WHITE ONLY",(0,1))
miscellaneous_milk=st.selectbox("MISCELLANEOUS MILK",(0,1))
plant=func_man_id(st.selectbox("manufacturer id",('1543', '926', '1078', '927', '5423', '1487', '1453', '531', '865','709')))
user_input = [retail_disc, coupon_disc, coupon_match_disc, day, volume, brand, fluid_white_milk, miscellaneous_milk]
for i in plant:
    user_input.append(plant[i])
user_input=np.array(user_input)
user_input=user_input.reshape(1,-1)
scaler = load('scaler_file.save')
user_input_scaled=scaler.transform(user_input)
st.write(user_input_scaled)
############

with open("spf_model.joblib", "rb") as f:
    clf = load(f)
    # make predictions
    prediction = clf.predict(user_input_scaled)
    st.header(f"The model predicts: {prediction[0]}")

