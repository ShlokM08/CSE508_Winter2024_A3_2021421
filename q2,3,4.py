# import pandas as pd
# from bs4 import BeautifulSoup
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

# # Assuming NLTK resources are set up as before

# def preprocess_text(text):
#     """Preprocesses review text."""
#     text = BeautifulSoup(text, "html.parser").get_text()
#     text = re.sub("[^a-zA-Z]", " ", text).lower()
#     words = word_tokenize(text)
#     words = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stopwords.words('english')]
#     return " ".join(words)

# # This function remains unchanged, provided earlier
# def load_and_process_in_chunks(reviews_path, metadata_path, product_keyword):
#     # Code as provided earlier...
#     return processed_data

# # Function to process reviews directly from the dataframe
# def process_reviews(preprocessed_data, metadata_df):
#     """Processes reviews from preprocessed data directly."""
#     # Filter metadata for "Headphones"
#     headphones_metadata = metadata_df[metadata_df['title'].str.contains('Headphone', case=False, na=False)]
#     filtered_asins = set(headphones_metadata['asin'].unique())
    
#     # Directly filter the preprocessed data without needing to rechunk
#     filtered_data = preprocessed_data[preprocessed_data['asin'].isin(filtered_asins)]

#     # Aggregate statistics
#     total_reviews = len(filtered_data)
#     average_rating = filtered_data['overall'].mean()
#     unique_products = filtered_data['asin'].nunique()
#     good_ratings_count = filtered_data[filtered_data['overall'] > 3].shape[0]
#     bad_ratings_count = filtered_data[filtered_data['overall'] <= 3].shape[0]
#     ratings_per_rating = filtered_data['overall'].value_counts().sort_index()

#     # Output the statistics
#     print(f"Number of Reviews: {total_reviews}")
#     print(f"Average Rating Score: {average_rating}")
#     print(f"Number of Unique Products: {unique_products}")
#     print(f"Number of Good Ratings: {good_ratings_count}")
#     print(f"Number of Bad Ratings: {bad_ratings_count}")
#     print("Reviews per Rating:")
#     print(ratings_per_rating)

# # Set file paths for reviews and metadata
# reviews_path = 'C:\\Users\\Shlok Mehroliya\\Downloads\\IR_assignment3\\Electronics_5.json.gz'
# metadata_path = 'C:\\Users\\Shlok Mehroliya\\Downloads\\IR_assignment3\\meta_Electronics.json.gz'

# # Load and process data
# processed_data = load_and_process_in_chunks(reviews_path, metadata_path, 'Headphone')

# # Assuming the metadata has been loaded
# metadata_df = pd.read_json(metadata_path, lines=True, compression='gzip')

# # Now process the reviews with the processed data and metadata
# process_reviews(processed_data, metadata_df)
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from q1 import *

def load_processed_data(file_path):
    """Loads the processed data from a file."""
    return pd.read_csv(file_path)

def analyze_data(df):
    """Performs data analysis and visualization."""
    # Report total number of rows
    print(f"Total number of rows: {len(df)}")
    
    # Handle missing values and duplicates
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['processed_reviewText'], inplace=True)
    
    # Descriptive Statistics
    print(f"Number of Reviews: {df.shape[0]}")
    print(f"Average Rating Score: {df['overall'].mean()}")
    print(f"Number of Unique Products: {df['asin'].nunique()}")
    df['rating_category'] = np.where(df['overall'] >= 3, 'Good', 'Bad')
    print(f"Number of Good Ratings: {df[df['rating_category'] == 'Good'].shape[0]}")
    print(f"Number of Bad Ratings: {df[df['rating_category'] == 'Bad'].shape[0]}")
    
    # Number of Reviews corresponding to each Rating
    print(df['overall'].value_counts().sort_index())
    
    # Visualization Example: Word Cloud for Good Ratings
    good_reviews_text = " ".join(review for review in df[df['rating_category'] == 'Good']['processed_reviewText'])
    wordcloud_good = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(good_reviews_text)
    plt.figure()
    plt.imshow(wordcloud_good, interpolation="bilinear")
    plt.axis("off")
    plt.title("Good Ratings Word Cloud")
    plt.show()
    # Further analysis and visualizations as required

if __name__ == "__main__":
    processed_file_path = 'processed_data.csv'  # Ensure this matches the filename from q1.py
    processed_data = load_processed_data(processed_file_path)
    analyze_data(processed_data)
