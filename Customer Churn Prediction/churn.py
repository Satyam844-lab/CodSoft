import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df= pd.read_csv("Churn_Modelling.csv")
#dropping not useful columns 
df=df.drop(['RowNumber','CustomerId','Surname',],axis =1)
#one-hot encoding 
df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)

#scaling numeric values 
scaler=StandardScaler()
num_cols= ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember','EstimatedSalary']
df[num_cols]=scaler.fit_transform(df[num_cols])

#seperate train and test data 
x=df.drop('Exited',axis=1)
y=df['Exited']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train model
model=LogisticRegression(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#checking metrices
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Get user inputs for a new customer
def get_user_input_and_predict():
    print("\nEnter customer details to get churn prediction!")

    credit_score = float(input("Credit Score: "))
    age = float(input("Age: "))
    tenure = float(input("Tenure (years as customer): "))
    balance = float(input("Balance ($): "))
    num_of_products = float(input("Number of Products: "))
    has_cr_card = int(input("Has Credit Card? (1 = Yes, 0 = No): "))
    is_active_member = int(input("Is Active Member? (1 = Yes, 0 = No): "))
    est_salary = float(input("Estimated Salary ($): "))
    
    geo = input("Geography (France, Germany, Spain): ").strip().lower()
    geo_germany = 1 if geo == "germany" else 0
    geo_spain   = 1 if geo == "spain" else 0
    
    gender_str = input("Gender (Male, Female): ").strip().lower()
    gender_male = 1 if gender_str == "male" else 0
    
    # Prepare input as DataFrame, scale numerical values
    user_array = np.array([credit_score, age, tenure, balance, num_of_products,  has_cr_card, is_active_member, est_salary, geo_germany, geo_spain, gender_male]).reshape(1, -1)
    input_cols=list(x.columns)
    user_df=pd.DataFrame(user_array,columns=input_cols)
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] 
    user_df[num_cols] = scaler.transform(user_df[num_cols])
    
    # Make prediction
    pred = model.predict(user_df)
    print("\nPrediction:")
    print("Likely to churn." if pred[0] == 1 else "Likely to stay.")
   
get_user_input_and_predict()
