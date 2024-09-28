# TASK-1
# Iris Flower Classification
# Iris Flower Species Classification

## Project Overview
This project classifies iris flower species based on measurements of sepal length, sepal width, petal length, and petal width. The project uses several machine learning models, compares their accuracy, and deploys the best-performing model using Gradio for a user-friendly interface. The dataset used is the Iris dataset, a well-known dataset for pattern recognition.

## Dataset Description
The Iris dataset contains 150 instances of iris flowers belonging to three species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*. Each instance provides measurements of four features:

- **Sepal length (cm)**
- **Sepal width (cm)**
- **Petal length (cm)**
- **Petal width (cm)**

The dataset is balanced, with 50 samples for each species.

## Objective
The goal is to build a classification model that accurately predicts the species of an iris flower based on the provided measurements. The model is trained on historical data, and we compare the performance of multiple algorithms before selecting the best one for deployment.

## Steps

### 1. Importing Dependencies
We import essential libraries like `pandas` for data manipulation, `numpy` for numerical computations, and machine learning models from `scikit-learn`. Gradio is used to create a simple web interface to interact with the trained model.

### 2. Data Collection
We load the Iris dataset using `pandas` and display the first few records to understand the structure of the data. The dataset contains 150 rows and 6 columns. The features include four numerical columns (sepal and petal measurements), and the target variable is the species of the iris flower.

### 3. Splitting Data into Training and Testing Sets
To evaluate our model’s performance, we split the dataset into a training set (80%) and a testing set (20%) using the `train_test_split` function from `scikit-learn`. The model is trained on the training set and evaluated on the testing set.

### 4. Initializing Classifiers
We initialize various classification models:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Naive Bayes**

These models are popular for classification tasks, and we aim to identify the most accurate one by comparing their performance.

### 5. Model Training and Evaluation
Each classifier is trained on the training dataset, and predictions are made on the testing dataset. We measure the model's accuracy using the `accuracy_score` metric from `scikit-learn`. The accuracy results are printed, and the **K-Nearest Neighbors (KNN)** model is selected as the most accurate, achieving an accuracy of 100%.

### 6. Testing the Model
After selecting the KNN model, we evaluate its performance by making predictions on a test case with sepal and petal measurements (5.8, 2.8, 5.1, 2.4). The model correctly predicts the species as *Iris-virginica*.

### 7. Deploying the Model with Gradio
We create a simple web interface using Gradio to allow users to input sepal and petal measurements and receive predictions about the species of the iris flower. The Gradio interface has four input fields (for the flower measurements) and a text output that displays the predicted species.

## Machine Learning Algorithms Used

### K-Nearest Neighbors (KNN)
KNN is a non-parametric, instance-based learning algorithm. It works by finding the k nearest neighbors to a given data point and assigning the most common class among the neighbors as the prediction. KNN is known for its simplicity and effectiveness, especially in small datasets like Iris.

### Logistic Regression
Logistic regression is a linear model used for binary and multiclass classification. It predicts the probability that a given data point belongs to a certain class by fitting a logistic curve.

### Support Vector Machine (SVM)
SVM is a classification technique that finds the hyperplane that best separates different classes in the feature space. SVM is effective for high-dimensional data and cases where the decision boundary is not linear.

### Decision Tree
A decision tree is a tree-like structure where each node represents a feature, and branches represent possible feature values. It makes decisions by splitting data into subsets based on the feature that maximizes information gain.

### Random Forest
Random Forest is an ensemble learning method that builds multiple decision trees and merges them to produce more accurate and stable predictions.

### Naive Bayes
Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with the assumption of independence between features. Despite the simplistic assumption, it performs well in many complex real-world scenarios.

## Gradio Interface
Gradio is a Python library that allows you to quickly create customizable UIs for machine learning models. In this project, we used Gradio to deploy the KNN model, making it accessible for anyone to input iris flower measurements and get a prediction about the species.

## Conclusion
This project demonstrates the use of machine learning algorithms to classify iris flower species. After training and comparing various models, we deployed the best-performing model (KNN) using Gradio for easy interaction. The project provides a foundation for understanding classification models and web deployment.

