import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

import joblib
model = joblib.load('lgb_clfer_model.ml')

df = pd.read_csv(r'C:\Users\victo\Desktop\kaggle\bank_churn\BankChurners.csv')
df.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                  'CLIENTNUM'],inplace=True)
#changing target value to int
churn_map = {"Existing Customer": 0, "Attrited Customer":1}
df.replace({"Attrition_Flag": churn_map},inplace=True)
df.rename(columns = {'Attrition_Flag': 'y'},inplace=True)
cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
col_transformer = make_column_transformer((OneHotEncoder(), cols), remainder='passthrough')
pipeline = Pipeline([('preprocess',col_transformer)])


#Preparing data for training
y = df['y']
X = df.drop(columns='y')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42,stratify=y)
x_test_pro = pipeline.fit_transform(x_test)

y_pred = model.predict(x_test_pro)
from sklearn.metrics import classification_report
print(classification_report((y_pred),y_test))
#model.predict(x_test)