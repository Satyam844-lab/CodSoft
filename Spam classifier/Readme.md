SMS Spam Detection using TF-IDF and Naive Bayes

This project is a simple machine learning model to classify SMS messages as spam or not spam (ham).

It uses "TF-IDF vectorization" to convert text messages into numerical form and a Naive Bayes classifier to predict whether a message is spam.

Features
Text preprocessing using TF-IDF
Spam/Ham prediction using Multinomial Naive Bayes
Interactive command-line interface to test your own messages
Evaluation metrics: accuracy, confusion matrix, classification report
Dataset

The dataset used is the SMS Spam Collection dataset, which contains 5,572 SMS messages labeled as spam or ham.

*How it works

1️⃣ The dataset is loaded and preprocessed.
2️⃣ Text data is vectorized using TF-IDF (TfidfVectorizer).
3️⃣ A Naive Bayes classifier (MultinomialNB) is trained on the vectorized text.
4️⃣ The model is evaluated on a test set (accuracy and other metrics printed).
5️⃣ You can enter your own message and see live predictions
