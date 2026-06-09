import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the Titanic dataset
passenger_data = pd.read_csv("titanic.csv")

# Fill missing values in Age with median value
passenger_data["Age"].fillna(passenger_data["Age"].median(), inplace=True)

# Fill missing values in Embarked with most frequent value
passenger_data["Embarked"].fillna(passenger_data["Embarked"].mode()[0], inplace=True)

# Convert categorical data into numeric format
passenger_data["Sex"] = passenger_data["Sex"].replace({"male": 0, "female": 1})
passenger_data["Embarked"] = passenger_data["Embarked"].replace({"S": 0, "C": 1, "Q": 2})

# Choose the columns that will be used for prediction
input_data = passenger_data[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

# Survival column is the target
output_data = passenger_data["Survived"]

# Split the dataset into training and testing parts
train_features, test_features, train_labels, test_labels = train_test_split(
    input_data,
    output_data,
    test_size=0.20,
    random_state=42
)

# Create and train the Random Forest model
survival_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

survival_model.fit(train_features, train_labels)

# Predict survival on test data
predicted_result = survival_model.predict(test_features)

# Calculate accuracy
final_accuracy = accuracy_score(test_labels, predicted_result)

print("Titanic Survival Prediction")
print("-" * 35)
print("Model Accuracy :", final_accuracy)

print("\nClassification Report:\n")
print(classification_report(test_labels, predicted_result))