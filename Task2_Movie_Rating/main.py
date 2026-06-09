import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the movie dataset
movie_data = pd.read_csv("movie.csv", encoding="latin1")

# Remove rows with missing values
movie_data = movie_data.dropna()

# Create label encoder
label_encoder = LabelEncoder()

# Convert categorical columns into numeric values
movie_data["Genre"] = label_encoder.fit_transform(movie_data["Genre"])
movie_data["Director"] = label_encoder.fit_transform(movie_data["Director"])
movie_data["Actor 1"] = label_encoder.fit_transform(movie_data["Actor 1"])
movie_data["Actor 2"] = label_encoder.fit_transform(movie_data["Actor 2"])
movie_data["Actor 3"] = label_encoder.fit_transform(movie_data["Actor 3"])

# Clean Year column
movie_data["Year"] = (
    movie_data["Year"]
    .astype(str)
    .str.extract("(\d{4})")[0]
)

movie_data["Year"] = pd.to_numeric(movie_data["Year"], errors="coerce")

# Clean Votes column
movie_data["Votes"] = (
    movie_data["Votes"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

movie_data["Votes"] = pd.to_numeric(movie_data["Votes"], errors="coerce")

# Remove remaining missing values
movie_data = movie_data.dropna()

# Select input features
input_features = movie_data[
    [
        "Year",
        "Genre",
        "Votes",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3",
    ]
]

# Target column
movie_rating = movie_data["Rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    input_features,
    movie_rating,
    test_size=0.20,
    random_state=42,
)

# Create and train model
rating_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)

rating_model.fit(X_train, y_train)

# Predict ratings
predicted_rating = rating_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predicted_rating)
r2 = r2_score(y_test, predicted_rating)

print("\nMovie Rating Prediction")
print("-" * 40)
print("Mean Absolute Error :", round(mae, 2))
print("R2 Score :", round(r2, 2))

# Show comparison table
comparison = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": predicted_rating
})

print("\nFirst 10 Predictions:\n")
print(comparison.head(10))

# Plot graph
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predicted_rating,
    alpha=0.6,
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")

plt.grid(True)

# Save graph
plt.savefig("movie_rating_graph.png")

plt.show()