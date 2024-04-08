import pandas as pd  # Corrected import statement for pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
from sklearn.preprocessing import LabelEncoder
import numpy as np
from q5 import *

df = pd.read_csv('your_preprocessed_data.csv')  # Make sure to adjust this path


def categorize_rating(overall):
    if overall > 3:
        return 'Good'
    elif overall == 3:
        return 'Average'
    else:
        return 'Bad'

df['rating_class'] = df['overall'].apply(categorize_rating)


tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_reviewText'])

# Encode the rating class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['rating_class'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.25, random_state=42)

# Save the train and test datasets
save_npz('X_train.npz', X_train)
save_npz('X_test.npz', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Datasets saved successfully.")
