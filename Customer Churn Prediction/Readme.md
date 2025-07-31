<h1>🏦 Bank Customer Churn Prediction</h1>

<p>A machine learning project to predict which bank customers are likely to leave ("churn") using logistic regression on real customer data. This repository includes all steps from data preparation to evaluation and a user-friendly way to enter new customer details for live predictions.</p>

<hr>

<h2>🚀 Project Overview</h2>
<ul>
  <li><strong>Problem:</strong> Customer churn is costly for businesses. Predicting who is likely to churn helps retain valuable customers through targeted interventions.</li>
  <li><strong>Objective:</strong> Use historical customer data to build a model that predicts churn, allowing for early retention efforts.</li>
  <li><strong>Approach:</strong> Step-by-step workflow using Python, pandas, and scikit-learn, focused on clarity and learning.</li>
</ul>

<hr>

<h2>📂 Dataset</h2>
<ul>
  <li><strong>Name:</strong> Bank Customer Churn Prediction (from Kaggle)</li>
  <li><strong>Link:</strong> <a href="https://www.kaggle.com/datasets/shubhendra7/bank-customer-churn-prediction">Bank Customer Churn</a></li>
  <li><strong>Rows:</strong> 10,000</li>
  <li><strong>Features:</strong> Demographic, financial, and behavioral attributes</li>
  <li><strong>Target:</strong> <code>Exited</code> (1 = churned/left, 0 = stayed)</li>
  <li><strong>Key Features:</strong>
    <ul>
      <li>CreditScore</li>
      <li>Age</li>
      <li>Tenure</li>
      <li>Balance</li>
      <li>NumOfProducts</li>
      <li>HasCrCard</li>
      <li>IsActiveMember</li>
      <li>EstimatedSalary</li>
      <li>Geography</li>
      <li>Gender</li>
    </ul>
  </li>
</ul>

<hr>

<h2>🛠️ Workflow & Methods</h2>
<ul>
  <li><strong>Data Cleaning:</strong> Dropped irrelevant columns: RowNumber, CustomerId, Surname</li>
  <li><strong>Exploratory Data Analysis:</strong> Inspected missing values, got data overview</li>
  <li><strong>Data Preprocessing:</strong>
    <ul>
      <li>One-hot encoded categorical features (Geography, Gender)</li>
      <li>Scaled numerical features for uniformity</li>
    </ul>
  </li>
  <li><strong>Modeling:</strong>
    <ul>
      <li>Split data into train/test sets (80/20 split)</li>
      <li>Trained logistic regression for interpretability and baseline performance</li>
    </ul>
  </li>
  <li><strong>Evaluation:</strong>
    <ul>
      <li>Accuracy, precision, recall, F1-score, confusion matrix</li>
      <li>Model is strong at identifying non-churners, less so at catching actual churners (common for simple models on imbalanced data)</li>
    </ul>
  </li>
  <li><strong>User Interaction:</strong> Built an interactive function to input new customer data and predict churn likelihood in real time</li>
</ul>

<hr>

<h2>📈 Results</h2>
<ul>
  <li><strong>Accuracy:</strong> ~81%</li>
  <li><strong>Recall (for churners):</strong> ~20% (model catches some, but not all true churners—typical for this dataset and simple models)</li>
  <li><strong>Business Insight:</strong> Customers with higher account activity, more products, or larger balances are less likely to churn, while certain demographics or low engagement raise risk.</li>
</ul>

<hr>

<h2>📝 Limitations & Future Improvements</h2>
<ul>
  <li>Logistic regression gives a strong baseline but struggles with imbalanced data (low recall for churners).</li>
  <li><strong>Possible improvements:</strong>
    <ul>
      <li>Use advanced models (Random Forest, XGBoost)</li>
      <li>Balance the dataset or adjust class weights</li>
      <li>Feature engineering</li>
      <li>Build a GUI or deploy as a web app</li>
    </ul>
  </li>
</ul>
