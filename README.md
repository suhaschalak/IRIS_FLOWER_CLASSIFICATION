# Iris Flower Species Classification

## Project Description
Classify the iris flower species from the measurements of sepal length and width, along with petal length and width. The author tries out a suite of several machine learning models, compares their accuracies, and finally deploys the best one by using Gradio to make it readily accessible to a user. The dataset applied here is the classic Iris dataset, which has long been used in pattern recognition.

## Dataset Description
![Iris Flower Species](![image](https://github.com/user-attachments/assets/e559f006-b1d7-482b-8d67-e77ad569bcfa))
The Iris dataset features 150 instances of iris flowers, thus belonging to three different species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*. Each instance provides measurements for four features:
 
- Sepal length in cm
- Sepal width in cm
- Petal length in cm
- Petal width in cm

The database is balanced because it contains 50 samples from each species.

## Goal
We will try to design an accurate classifier that can predict the species given measurements of an iris flower. The model trains on historical data and compares a few different algorithms before we pick the best one to implement.

## Steps

### 1. Importing Dependencies
We then import needed libraries, namely `pandas` for data manipulation, `numpy` for numerical computation, and machine learning models from `scikit-learn`. We further create a simple web interface using Gradio to interact with the trained model.
 
### 2. Data Collection
We import Iris data using `pandas`, and preview a few lines in order to get a feel for the structure of the data. The dataset has 150 rows by 6 columns. Four columns are numeric: sepal and petal measures, and the target variable is species of the iris.
### 3. Data Splitting Train / Test Sets
We use the `train_test_split` function from `scikit-learn` to split the dataset into a training set with 80% and a testing set with 20%. We then train our model on the training set and run it on the testing set.

### 4. Instantiating Classifiers
We import the following classification models, initializing them below:

-   Logistic Regression
-   Support Vector Machine (SVM)
-   K-Nearest Neighbors (KNN)
-   Decision Tree
-   Random Forest
-   Naive Bayes

We use these models extensively in classification tasks, and hence we want to know which one gives the most accurate prediction of any. Let us then proceed to compare their performance.

### 5. Training and Evaluation Models
Each classifier is trained on the training dataset and used to make predictions on the testing dataset. We evaluate the accuracy of the model using the `accuracy_score` metric of `scikit-learn`. The accuracy is printed, and the **K-Nearest Neighbors (KNN)** is the model with the highest accuracy at 100%.

### 6. Testing the Model
We now select the KNN model and use it on a test case with sepal and petal measurements of (5.8, 2.8, 5.1, 2.4). The model correctly classifies the species as *Iris-virginica*.

### 7. Deploying the Model with Gradio
We design a simple web interface using Gradio such that a user can enter measurements of dimensions of sepal and petal and then get predictions for the species of iris flower. The Gradio interface comes with four input fields, which will be for the flower measurements, and a text field which will write out what it predicts about the species.

## Machine Learning Algorithms Used

### K-Nearest Neighbors (KNN)
KNN is a non-parametric, instance-based learning algorithm which works by finding the k nearest neighbors of a given data point and returning the most frequent class of the neighbors as the prediction. KNN is easy to use and normally pretty effective, such as when dealing with a small dataset like Iris.

### Logistic Regression
Logistic Regression is actually linear and used for both binary and multiclass classifications. It predicts the probability of a given point belonging to a certain class by fitting a logistic curve.

### Support Vector Machine (SVM)
SVM is a classification technique that finds the separation hyperplane that best classifies different classes in the feature space. SVM is efficient when the feature dimension is high and the decision boundary is complex.

### Decision Tree
A decision tree is a tree-like structure where each node is represented as a feature, and branches represent possible feature values. It takes decisions by splitting data into subsets based on the feature that maximizes information gain.

### Random Forest
Random Forest is an ensemble learning method that constructs multiple decision trees and merges them to provide more accurate and stable predictions.

### Naive Bayes
Naive Bayes is one kind of probabilistic classifier, which relies on Bayes' theorem under the assumption of independence between features. It surprisingly works quite well in many complicated applications with real-world data.

## Gradio Interface
This is a Python library that enables you to build customisable UIs for machine learning models. In our project, we used Gradio to deploy the KNN model such that anyone can input iris flower measurements and see what kind of prediction will be obtained.
## Conclusion
This is an application that demonstrates classifying iris flower species using machine learning algorithms. Based on the training and comparison of models, I proceeded to web deployment using the Gradio interface to make the KNN model available for easy interaction. The project provides a foundation for understanding classification models and web deployment.
