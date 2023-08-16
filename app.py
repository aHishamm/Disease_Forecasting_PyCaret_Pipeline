import pandas as pd 
import numpy as np 
import plotly.express as px
import seaborn as sns
from pycaret.time_series import TSForecastingExperiment
import streamlit as st 
from io import StringIO

def filter_df(data_frame: pd.DataFrame,item_list: list) -> pd.DataFrame: 
    return df[df['Entity'].isin(item_list)] 

st.title("Children Cause of Death Dashboard")
uploaded_csv = st.file_uploader("Upload a File:")
if uploaded_csv is not None: 
    df = pd.read_csv(uploaded_csv)
    df.drop(columns=['Code'],inplace=True,errors='ignore')
    disease_list = [i.split(' - ')[1] for i in (list(df.columns))[2:]]
    entity_list = list(df['Entity'].unique())
    df.columns = ['Entity','Year'] + disease_list
    df_list = [df[df['Entity'] == i] for i in list(df['Entity'].unique())]
    st.subheader("Dataframe Preview:")
    st.dataframe(df.head(20))
    entity_multiselect = st.multiselect('Choose one or more countries: ',entity_list) 
    disease_selectbox = st.selectbox('Choose the disease: ',disease_list) 
    if st.button("Process"): 
        f_df = filter_df(df,entity_multiselect) 
        fig = px.line(f_df,x='Year',y=disease_selectbox,color='Entity')
        st.plotly_chart(fig)
    ml_entity_select = st.selectbox('Select a country: ',entity_list,key=2) 
    ml_disease_select = st.selectbox('Select the disease: ',disease_list,key=3)  
    if st.button("Process",key=4,):
        ml_df = filter_df(df,[ml_entity_select])
        ml_df = ml_df[['Year',ml_disease_select]] 
        st.dataframe(ml_df)
        s = TSForecastingExperiment() 
        fh = 5 
        s.setup(ml_df,fh = fh, target = ml_disease_select, session_id = 123)
        #Elastic Net w/ Cond. Deseasonalize & Detrending
        best = s.create_model(estimator='en_cds_dt')
        max_year = ml_df['Year'].max() + 1 
        num_list = [max_year + i for i in range(fh)]
        forecast_df = pd.DataFrame({'Year':num_list,
                                   ml_disease_select:list(s.predict_model(best)['y_pred'])})
        print(forecast_df)
        ml_df = pd.concat([ml_df, forecast_df], ignore_index=True)
        print(ml_df)
        st.subheader(f'Predictive Analysis for {ml_disease_select} in {ml_entity_select} for the next 5 years')
        fig = px.line(ml_df,x='Year',y=ml_disease_select)
        st.plotly_chart(fig)








