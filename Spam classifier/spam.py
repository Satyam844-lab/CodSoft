import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv(r"C:\Users\satya\Desktop\ml pro\spam.csv",encoding='latin1')
df['v1']=df['v1'].map({'spam':1,'ham':0})

x=df['v2']
y=df['v1']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

tfidf=TfidfVectorizer()
x_train_vector=tfidf.fit_transform(x_train)
x_test_vector=tfidf.transform(x_test)

model=MultinomialNB()
model.fit(x_train_vector,y_train)

y_pred=model.predict(x_test_vector)
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print(f"accuracy:{accuracy*100:.2f} \n confusion matrix :{confusion}  \n report:{report}")

while True:
     message = input("Enter your message (or type 'exit' to quit): ")
     if message.lower() == "exit":
         break
     message_vector = tfidf.transform([message])
     pred = model.predict(message_vector)
     print("Prediction:", "spam ❌" if pred[0] == 1 else "not spam ✅")

