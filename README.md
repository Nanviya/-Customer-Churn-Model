# **Customer Churn Prediction Model**

This project uses machine learning to predict customer churn for a telecommunications company. The model is trained on a dataset containing various customer attributes, including demographic details, account information, and service usage. The primary objective is to predict whether a customer will churn (leave the service) or not, enabling the business to proactively take steps to retain them.

---

### **📋 Table of Contents**
* [Dataset](#-dataset)
* [Methodology](#-methodology)
* [Model Performance](#-model-performance)
* [File Structure](#-file-structure)
* [Setup and Installation](#-setup-and-installation)
* [How to Use the Predictive System](#-how-to-use-the-predictive-system)
* [Contact](#-contact)

---

### **📊 Dataset**

The model is trained on the **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). This dataset contains information about 7,043 customers and includes features such as:

* **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Information**: `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Services Subscribed**: `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.

---

### **⚙️ Methodology**

The project follows a standard machine learning workflow:

1.  **Data Cleaning**: The initial dataset was loaded and cleaned. The `customerID` column was dropped, and missing values in the `TotalCharges` column were handled.

2.  **Exploratory Data Analysis (EDA)**: Visualizations like count plots, histograms, and a correlation heatmap were used to understand the distribution of features and their relationships.

3.  **Data Preprocessing**:
    * **Encoding**: All categorical features were converted into numerical format using `sklearn.preprocessing.LabelEncoder`. The target variable `Churn` was manually encoded (`Yes`: 1, `No`: 0).
    * **Handling Class Imbalance**: The training data showed a significant class imbalance. To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training set to create a balanced distribution for the model.

4.  **Model Training**:
    * Several classification models (Decision Tree, Random Forest, XGBoost) were evaluated using 5-fold cross-validation.
    * The **Random Forest Classifier** was selected as the final model due to its higher cross-validation accuracy of **84%**.

5.  **Model Saving**: The trained Random Forest model and the fitted label encoders were saved as `customer_churn_model.pkl` and `encoders.pkl` respectively.

---
### **📈 Model Performance**

The final Random Forest model was evaluated on the unseen test set to measure its real-world performance.

**IMPORTANT:** Replace the scores below with the actual results from the `classification_report` you ran in your notebook.

| Metric         | Score      |
| -------------- | ---------- |
| **Accuracy** | **81%** |
| **Precision** | **65%** (Churn) |
| **Recall** | **70%** (Churn) |
| **F1-Score** | **67%** (Churn) |


---

### **📂 File Structure**

├── Customer_Churn.ipynb                 # Jupyter Notebook with the full analysis.
├── customer_churn_model.pkl             # Saved file for the trained Random Forest model.
├── encoders.pkl                         # Saved file for the categorical feature encoders.
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # The dataset file.
└── README.md                            # This documentation file.


---

---

### **🚀 Setup and Installation**

To run this project locally, clone the repository and install the necessary dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/Nanviya/Customer-Churn-Model>
    cd <Customer-Churn-Model>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
    ```

---

### **💡 How to Use the Predictive System**

The saved model can be used to predict churn for a new customer.

**Example Python Script:**
```python
import pandas as pd
import pickle

# Load the trained model and the encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Sample new customer data
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Create a DataFrame and encode the features
input_data_df = pd.DataFrame([input_data])
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

# Print the results
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability (No Churn, Churn): {pred_prob}")

Sample Output:

Prediction: No Churn
Prediction Probability (No Churn, Churn): [[0.78 0.22]]

📬 Contact
Nanviya Zala – nanviyazala1234@gmail.com

Project Link: https://github.com/Nanviya/Customer-Churn-Model

