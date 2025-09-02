# yo chai ho PCA_clustering of salary data ko ho 
import pandas as pd
import plotly as pt
import pickle as pk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.write("PCA_Clustering of salary data")

load_model = pk.load(open("salary_clusterings.pickle", 'rb'))
# st.write("Model expects features:", load_model.feature_names_in_)


age = st.number_input("enter your age",18,60)  # age yeti dekhi yeti bhnna rakheko
exp = st.number_input("Enter your years of experience", 0,40)
salary = st.number_input("Enter your salary",350,250000)
# gender = st.radio("Gender = ", ["Male","Female"])
edu = st.selectbox("Education =", ["Bachelor", "Master", "PHD"])

# if gender =='Male':
#     male = True
# else:
#     male = False

if edu == "Bachelor":
    b = 1; m = 0; p = 0
elif edu == "Master":
    b =0; m = 1; p = 0
else:
    b = 0; m= 0; p =1

if st.button("predict"):
   df = pd.DataFrame({
      'Age':[age],
      'Years of Experience':[exp],
      'Salary':[salary],
    #   'Male':[male],   #yesma male ko thau ma gender rakhna mildaina kina ki mathi ko data frame ho hamley input dida chai true false garya thiyeu so true false wala nai variable rakhney
      "Bachelor's":[b],
      "Master's":[m],
      'PhD':[p]
   })
   st.dataframe(df)
   result = load_model.predict(df)
   st.balloons()
   if int(result.tolist()[0])== 0:
        st.write("you are in 0 cluster category")
   elif int(result.tolist()[0]) == 1:
        st.write("you are in 1 cluster category")
   elif int(result.tolist()[0]) == 2:
        st.write("you are in 2 cluster category")
   elif int(result.tolist()[0]) == 3:
        st.write("you are in 3 cluster category")
   elif int(result.tolist()[0]) == 4:
        st.write("you are in 4 cluster category")
   else:
         st.write("you are in 5th cluster category")


# simply maile PCA ani cluster ani transform yo tyo bhanda
# cluster ma paila x model banaye ra input data diye ani teslai nai model.fit(x)banaye ra pickle file banaye ani bhayo 

