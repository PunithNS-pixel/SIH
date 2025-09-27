import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (update with your CSV filename)
data = pd.read_csv('dataset1.csv')

# Preview dataset columns and data
print(data.head())

# Assume target column is 'fertility' (adjust based on dataset)
# Replace this with the actual target column name from the dataset
target_col = 'fertility'

# Features - drop target column
X = data.drop(target_col, axis=1)

# Target
y = data[target_col]

# Handle missing values - fill with mean
X.fillna(X.mean(), inplace=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train XGBoost Classifier
model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
