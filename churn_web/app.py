from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from helper_function import pre_process



app = Flask(__name__)



@app.route('/')
def home():
    return '<h1> Is work <h1>'
    #return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    model = joblib.load('lgb_clfer_model.ml')
    col_names = ['Customer_Age', 'Gender', 'Dependent_count',
       'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
       'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    data =[45, 'M', 3, 'High School', 'Married', 2, 'Blue', 39, 5, 1, 3,
           12691.0, 777, 11914.0, 1.335, 1144, 42, 1.625, 0.061]
    data = np.array(data)
    user_input = pd.DataFrame(data).T
    #user_input.rename(columns = col_names,inplace=True)
    col_map = dict(zip(np.arange(19), col_names))
    user_input.rename(columns = col_map,inplace=True)
    processed_data = pre_process(user_input)
    prediction = model.predict(processed_data)
    return str(prediction)


if __name__ == "__main__":
    app.run()
    
    
    
    
  
    