import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
def pre_process(user_input):
    '''
    col_names = ['Customer_Age', 'Gender', 'Dependent_count',
           'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
           'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    '''
    df = pd.read_csv('BankChurners.csv')
    df.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                          'CLIENTNUM'],inplace=True)
    churn_map = {"Existing Customer": 0, "Attrited Customer":1}
    Income_map = {'Less than $40K': '0','$40K - $60K':'1' , '$60K - $80K':'2', 
              '$80K - $120K':'3', '$120K +':'4', 'Unknown':'5'}
    df.replace({ "Attrition_Flag": churn_map,"Income_Category": Income_map},inplace=True)
    df.rename(columns = {'Attrition_Flag': 'y'},inplace=True)
    cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    col_transformer = make_column_transformer((OneHotEncoder(), cols), remainder='passthrough')
    pipeline = Pipeline([('preprocess',col_transformer)])
    df = df.drop(columns='y')
    pipeline.fit(df)
    
    float_data  = ['Credit_Limit', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
       'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    int64_data = ['Customer_Age', 'Dependent_count', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Total_Revolving_Bal', 'Total_Trans_Amt',
       'Total_Trans_Ct']
    user_input[int64_data] = user_input[int64_data].apply(pd.to_numeric)
    user_input[float_data] = user_input[float_data].apply(pd.to_numeric)
    clean_df = pipeline.transform(user_input)
    return clean_df

