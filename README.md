# Credit Card Churn Classifier


This repository was taken from Kaggle. The dataset was originally found on Leaps Analytica.

## 1. Summary 

##### Problem Statement

A manager at the bank is disturbed with more and more customers leaving their credit card services. This dataset has information on customers, and the goal is to provide them better services and turn customers' decisions in the opposite direction

##### Technical Overview
The dataset is slightly imbalanced, meaning there are only 15% of postitive cases (customers who churned). To overcome this imbalanced, I utilized ensemble models with SMOTE, a very robust ML technique that creates synthetic positive cases. 
## 2. Results

I tested 4 models:
- logistic regression with SMOTE
- Random Forest without SMOTE
- XGB with SMOTE
- LightGBM with SMOTE

Below is a plot of which features XGB thought was the most predictive feature. It seems like the total amount of transactions, and the difference in transactions are the most predictive indicators. 

![feat](https://github.com/victorvvu/Customer_Churn/blob/main/imgs/churn_feature.png?raw=true)


The best performing models are the XGBoost and LightGBM models while the the two models did not perform as well.
![ROC](https://github.com/victorvvu/Customer_Churn/blob/main/imgs/churn_roc.png?raw=true)


|Models| Precision | F1| Recall|
| :---         |     :---     |          :--- | :---  |  
| XGB with SMOTE  | .83   |  .89  |     .96|
| LGB with SMOTE | .83       | .89    | .96|
| Logistic Regression | .46 | .57 | .74|
|Random Forest | .73 | .82| .93|

One of the most important aspects of customer retention is identifying which customers are likely to churn. The model itself cannot prevent customers from churning; it is only a starting point. The next step would be to come up a strategy to incentivize at risk customers to stay with the company. This can include offer exclusive deals, more customer support, or offering bonuses from more spending. 
## 3. Data Description

This dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features. There are only 16.07% of customers who have churned.

- Attrition_Flag
- Customer_Age
- Gender
- Dependent_count
- Education_Level
- Marital_Status
- Income_Category
- Card_Category
- Months_on_book - how long customer has stayed
- Total_Relationship_Count
- Months_Inactive_12_mon
- Contacts_Count_12_mon
- Credit_Limit - credit card limit
- Total_Revolving_Bal - Balance
- Avg_Open_To_Buy - Open to Buy Credit Line (Average of last 12 months)
- Total_Amt_Chng_Q4_Q1 - amount change from Q4 - Q1
- Total_Trans_Amt - total amount of tran
- Total_Trans_Ct - total transaction count
- Total_Ct_Chng_Q4_Q1 - transaction count change from Q4 - Q1
- Avg_Utilization_Ratio - Average Card Utilization Ratio
  
## 4. Libraries

- Keras
- Optuna
- sklearn
- pandas
- numpy
- pandas 

## 5. References

https://www.kaggle.com/sakshigoyal7/credit-card-customers
