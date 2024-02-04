import streamlit as st
import xgboost
import pickle
import pandas as pd

pipe = pickle.load(open('Final_XGB_Model.pkl','rb'))



st.title('Car Price Predictor')

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age")
with col2:
    KM = st.number_input("KM")

Weight = st.number_input("Weight")

if st.button('Price Prediction'):
    input_df = pd.DataFrame({'Age': [Age],
                             'KM': [KM],
                             'Weight': [Weight]})
    st.table(input_df)


    def FunctionGeneratePrediction(inp_Age, inp_KM, inp_Weight):
        # Creating a data frame for the model input
        SampleInputData = pd.DataFrame(
            data=[[inp_Age, inp_KM, inp_Weight]],
            columns=['Age', 'KM', 'Weight'])
        # Calling the function defined above using the input parameters
        Predictions = FunctionPredictResult(InputData=SampleInputData)

        # Returning the predicted loan status
        return (Predictions.to_json())
    result = pipe.input_df