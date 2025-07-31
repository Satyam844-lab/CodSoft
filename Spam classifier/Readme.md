<h1>📧 SMS Spam Detection with Naive Bayes</h1>

<p>A machine learning project for classifying SMS messages as spam or not spam (ham), using Python, scikit-learn, and the popular <code>spam.csv</code> dataset. This project demonstrates key steps in building a real-world text classification model.</p>

<hr>

<h2>🚀 Project Overview</h2>
<ul>
  <li><strong>Goal:</strong> Automatically identify whether an SMS message is spam or not.</li>
  <li><strong>Approach:</strong> Preprocess text with TF-IDF, train a Naive Bayes classifier, and allow real-time message predictions from user input.</li>
</ul>

<hr>

<h2>📂 Dataset</h2>
<ul>
  <li><strong>Source:</strong> <a href="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset">Kaggle SMS Spam Collection Dataset</a></li>
  <li><strong>Size:</strong> ~5,000 labeled SMS messages</li>
  <li><strong>Columns:</strong>
    <ul>
      <li><code>v1</code>: Label (spam or ham)</li>
      <li><code>v2</code>: The message text</li>
    </ul>
  </li>
</ul>

<hr>

<h2>🛠️ Workflow</h2>
<ol>
  <li>Load Data: Read CSV, map spam/ham to 1/0 for modeling.</li>
  <li>Text Preprocessing: Use <code>TfidfVectorizer</code> to transform messages into numeric features.</li>
  <li>Split Data: Stratified train/test split for fair evaluation.</li>
  <li>Train Model: Multinomial Naive Bayes is well-suited for text data.</li>
  <li>Evaluate: Print accuracy, confusion matrix, and a full classification report.</li>
  <li>Predict New Messages: Interactive loop to test your own SMS strings for spam detection.</li>
</ol>

<hr>

<h2>📈 Results Example</h2>
<p><strong>Accuracy:</strong> 97.12%</p>

<p><strong>Confusion Matrix:</strong></p>
<pre>
[[1445    4]
 [  36  222]]
</pre>

<p><strong>Classification Report:</strong></p>
<pre>
              precision    recall  f1-score   support
           0       0.98      1.00      0.99      1449
           1       0.98      0.86      0.91       258
</pre>

<h4>🔍 Interpretation:</h4>
<ul>
  <li>Most real spam is caught (high recall for class <code>1</code>).</li>
  <li>Almost no real “ham” is classified as spam (high precision for class <code>0</code>).</li>
</ul>

<hr>

<h2>💻 How to Run</h2>

<h4>1. Clone the repo and install requirements:</h4>
<pre><code>pip install pandas scikit-learn</code></pre>

<h4>2. Get the Data:</h4>
<p>Download <code>spam.csv</code> from Kaggle and place it in your working directory.</p>

<h4>3. Run the script:</h4>
<pre><code>python spam_detector.py</code></pre>

<h4>4. Try your own messages:</h4>
<p>Enter an SMS string when prompted. Type <code>'exit'</code> to quit.</p>

<hr>

<h2>📝 How It Works</h2>
<ul>
  <li>SMS is vectorized using <strong>TF-IDF</strong> (turns words into numbers based on importance and frequency).</li>
  <li>The <strong>Naive Bayes classifier</strong> is trained to separate spam from legitimate messages using those features.</li>
  <li>In prediction mode, your message is vectorized and instantly classified as:
    <ul>
      <li><code>spam ❌</code></li>
      <li><code>not spam ✅</code></li>
    </ul>
  </li>
</ul>

<hr>

<h2>📖 Example Usage</h2>
<pre>
Enter your message (or type 'exit' to quit): 
Congratulations! Claim your FREE prize now.
Prediction: spam ❌

Enter your message (or type 'exit' to quit): 
Hey, are you coming to class?
Prediction: not spam ✅
</pre>

<hr>

<h2>📚 What I Learned</h2>
<ul>
  <li>How to preprocess text for ML using <strong>TF-IDF</strong>.</li>
  <li>Why <strong>Naive Bayes</strong> works well for document and SMS spam filtering.</li>
  <li>Importance of evaluation with <strong>confusion matrix</strong> and <strong>classification report</strong>.</li>
  <li>Simple way to build <strong>interactive prediction tools</strong>.</li>
</ul>

