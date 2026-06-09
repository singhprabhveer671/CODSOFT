import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris_data = pd.read_csv("iris.csv")

# Select input features and target column
features = iris_data.iloc[:, :-1]
target = iris_data.iloc[:, -1]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

# Create and train the classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on test data
predicted_species = classifier.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, predicted_species)

# Print the results
print("Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predicted_species))