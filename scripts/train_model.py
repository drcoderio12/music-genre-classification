import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the extracted features
features_file_path = 'C:/MIni Projects/DS/MGC Project/features/extracted_features.csv'

try:
    data = pd.read_csv(features_file_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    data = None  # Set data to None if loading fails

if data is not None:
    # Check for the presence of the label column
    if 'label' not in data.columns:
        print("Error: 'label' column not found in the dataset.")
        exit()

    X = data.iloc[:, :-1].values  # All feature columns
    y = data.iloc[:, -1].values    # Last column (labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save the trained model
    models_dir = 'C:/MIni Projects/DS/MGC Project/models'
    os.makedirs(models_dir, exist_ok=True)  # Ensure the models directory exists
    model_file_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(model, model_file_path)
    print(f'Model saved to {model_file_path}')
else:
    print("Data could not be loaded. Exiting the script.")
    exit()  # Exit if data is not loaded
