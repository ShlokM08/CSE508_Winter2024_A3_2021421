import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from q5 import *
import os

# Function to categorize ratings
def categorize_rating(overall):
    return 'Good' if overall > 3 else ('Average' if overall == 3 else 'Bad')

# Load the dataset
df = pd.read_csv('fully_preprocessed_data.csv')  # Adjust the file path

# Categorize the rating and prepare the data
df['rating_class'] = df['overall'].apply(categorize_rating)
X = df['processed_reviewText']
y = df['rating_class']

# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF Vectorization and splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Dimensionality Reduction to make models faster - Optimal for KNN
svd = TruncatedSVD(n_components=100, random_state=42)

# Define models within a pipeline for vectorization and optional dimensionality reduction
models = {
    "Logistic Regression": make_pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), LogisticRegression(max_iter=500)),
    "MultinomialNB": make_pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), MultinomialNB()),
    "LinearSVC": make_pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), LinearSVC(dual=False)),  
    "Decision Tree": make_pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), DecisionTreeClassifier(max_depth=10)),  
    "KNN with SVD": make_pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), svd, KNeighborsClassifier(n_neighbors=5))  
}


# Function to evaluate models
def evaluate_models(models, X_train, X_test, y_train, y_test, output_file='model_evaluation_reports.txt'):
    with open(output_file, 'w') as file:  # Open the file in write mode
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
            
            # Print to console
            print(f"Classification Report for {name}:")
            print(report)
            
            # Write to file
            file.write(f"Model: {name}\n")
            file.write(report + "\n\n")

# Evaluate models and save reports
evaluate_models(models, X_train, X_test, y_train, y_test)

