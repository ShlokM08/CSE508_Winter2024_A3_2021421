from q5 import  *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Load the preprocessed data
df = pd.read_csv('fully_preprocessed_data.csv')

# Initialize a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the 'processed_reviewText' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_reviewText'])

# Save the TF-IDF matrix to a file
# Note: For demonstration, saving as dense CSV, but sparse matrix formats are preferred for efficiency
dense_tfidf_matrix = tfidf_matrix.todense()
tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.to_csv('tfidf_matrix.csv', index=False)

# Alternatively, save the sparse TF-IDF matrix in a more efficient format
save_npz('tfidf_matrix.npz', tfidf_matrix)

print("TF-IDF matrix has been saved.")
