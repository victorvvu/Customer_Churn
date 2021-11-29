import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import joblib


#Preprocessing data
df = pd.read_csv(r'C:\Users\victo\Desktop\kaggle\bank_churn\BankChurners.csv')
df.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                  'CLIENTNUM'],inplace=True)
print(df.columns)
#changing target value to int / mapping income variable for later use
churn_map = {"Existing Customer": 0, "Attrited Customer":1}
Income_map = {'Less than $40K': '0','$40K - $60K':'1' , '$60K - $80K':'2', 
              '$80K - $120K':'3', '$120K +':'4', 'Unknown':'5'}

df.replace({ "Attrition_Flag": churn_map,"Income_Category": Income_map},inplace=True)

df.rename(columns = {'Attrition_Flag': 'y'},inplace=True)

#params found from optuna can be found in churn_model notebook
lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
              'boosting_type': 'gbdt','device_type': 'gpu', 'feature_pre_filter': False,
              'lambda_l1': 0.0, 'lambda_l2': 0.0, 'num_leaves': 17, 'feature_fraction': 0.5, 
              'bagging_fraction': 0.8230628101723434, 'bagging_freq': 5, 'min_child_samples': 20,'n_estimators':256}

lgb_sm_clfer = lgb.LGBMClassifier(**lgb_params)

#Pipeline for One hot encoding using sklearn
cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
col_transformer = make_column_transformer((OneHotEncoder(), cols), remainder='passthrough')
pipeline = Pipeline([('preprocess',col_transformer)])


#Preparing data for training
y = df['y']
X = df.drop(columns='y')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42,stratify=y)
pipeline.fit(x_train)

#Here we have the name of the variables used
preprocessed_col_names = pipeline[0].get_feature_names()
x_pro = pipeline.transform(x_train)

#Use SMOTE technique to create more data
SM = SMOTE(random_state=714)
x_SM, y_SM = SM.fit_resample(x_pro, y_train)
x_SM, x_sm_test, y_SM, y_sm_test = train_test_split(x_SM, y_SM, test_size=0.2, random_state=42,stratify=y_SM)

#fit model using SMOTE data and validate with SMOTE data, 
lgb_sm_clfer.fit(x_SM,y_SM, eval_set=[(x_sm_test,y_sm_test)],early_stopping_rounds=50, eval_metric="auc",)

#save model 
joblib.dump(lgb_sm_clfer, 'lgb_clfer_model.ml')
