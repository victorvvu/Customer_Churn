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
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = joblib.load('lgb_clfer_model.ml')
    col_names = ['Customer_Age', 'Gender', 'Dependent_count',
       'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
       'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    input1 = request.form['input1']
    input2 = request.form['input2']
    input3 = request.form['input3']
    input4 = request.form['input4']
    input5 = request.form['input5']
    input6 = request.form['input6']
    input7 = request.form['input7']
    input8 = request.form['input8']
    input9 = request.form['input9']
    input10 = request.form['input10']
    input11 = request.form['input11']
    input12 = request.form['input12']
    input13 = request.form['input13']
    input14 = request.form['input14']
    input15 = request.form['input15']
    input16 = request.form['input16']
    input17 = request.form['input17']
    input18 = request.form['input18']
    input19 = request.form['input19']
    
    data = [[input1,input2,input3,input4,input5,input6,input7,
             input8,input9,input10,input11,input12,input13,input14,
             input15,input16,input17,input18,input19]]
    
    data = np.array(data)
    user_input = pd.DataFrame(data, columns = col_names)
    
    #User helper function in helper_function.py
    processed_data = pre_process(user_input)
    prediction = model.predict(processed_data)
    if prediction == 0:
        return render_template('index2.html', prediction_text='The customer will not churn!')
    else:
        return render_template('index2.html', prediction_text='The customer will churn!')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    
    
    
    
  
    