import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Load test data
test_data = pd.read_csv("./data/processed/processed_test_zcore.csv")

# Load the pre-trained model
model = pickle.load(open("results/model.pkl", "rb"))

# Prepare test data
x_test = test_data.drop('Type', axis=1)
y_test = test_data['Type']

# Encode the target labels


# Make predictions
y_pred = model.predict(x_test)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)

# Choose the average method for precision and recall (for multiclass classification)
precision = precision_score(y_test, y_pred, average='weighted')  # 'micro', 'macro', 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # 'micro', 'macro', 'weighted'

# Store metrics in a dictionary
metrics_dict = {
    'accuracy': acc,
    'precision': precision,
    'recall': recall
}

# Save metrics to a JSON file
with open('reports/metrics_rfc.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)

print("Model evaluation completed successfully!")
