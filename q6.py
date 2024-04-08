import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from q5 import  *
# Function to save text outputs to a file
def save_text_to_file(filename, content):
    with open(filename, 'a') as file:
        file.write(content + "\n")

# Load the preprocessed data
df = pd.read_csv('fully_preprocessed_data.csv')

# Convert reviewTime to datetime format and extract year
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df['year'] = df['reviewTime'].dt.year

# File to store text outputs
output_file = 'eda_outputs.txt'

# Clear previous outputs
open(output_file, 'w').close()

# a. Top 20 most reviewed brands
most_reviewed_brands = df['brand'].value_counts().head(20).to_string()
print(most_reviewed_brands)
save_text_to_file(output_file, "Top 20 most reviewed brands:\n" + most_reviewed_brands)

# b. Top 20 least reviewed brands
least_reviewed_brands = df['brand'].value_counts().tail(20).to_string()
print(least_reviewed_brands)
save_text_to_file(output_file, "\nTop 20 least reviewed brands:\n" + least_reviewed_brands)

# c. The most positively reviewed ‘Headphone’
most_positive_review = df[df['overall'] > 3]['title'].value_counts().idxmax()
print(f"\nMost positively reviewed Headphone: {most_positive_review}")
save_text_to_file(output_file, f"\nMost positively reviewed Headphone: {most_positive_review}")

# d. Count of ratings over 5 consecutive years
ratings_by_year = df[df['year'].between(df['year'].min(), df['year'].min() + 4)]['year'].value_counts().sort_index().to_string()
print(ratings_by_year)
save_text_to_file(output_file, "\nCount of ratings over 5 consecutive years:\n" + ratings_by_year)
