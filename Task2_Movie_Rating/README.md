# Movie Rating Prediction with Python

## Introduction

The objective of this project is to predict movie ratings using machine learning techniques. The model analyzes different movie attributes such as genre, release year, director, actors, and number of votes to estimate the rating of a movie.

## Dataset

The dataset contains information about movies, including:

* Movie Name
* Release Year
* Genre
* Director
* Actor 1
* Actor 2
* Actor 3
* Votes
* Rating

The **Rating** column is used as the target variable for prediction.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

## Machine Learning Algorithm

Random Forest Regressor

## Project Workflow

* Loaded the movie dataset.
* Removed missing values from the dataset.
* Converted categorical columns into numerical values using Label Encoding.
* Cleaned and prepared the Year and Votes columns.
* Selected important features for model training.
* Split the dataset into training and testing sets.
* Trained the Random Forest Regression model.
* Predicted movie ratings on test data.
* Evaluated the model using Mean Absolute Error (MAE) and R² Score.
* Generated a graph comparing actual and predicted movie ratings.

## Output

The model predicts movie ratings based on the given features and displays performance metrics along with a comparison graph of actual and predicted ratings.

## Conclusion

This project demonstrates how machine learning can be used to predict movie ratings using historical movie data. It provides practical experience in data preprocessing, feature engineering, model training, prediction, and visualization using Python.
