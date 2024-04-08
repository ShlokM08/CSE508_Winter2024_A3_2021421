import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Assuming q5.py contains necessary preprocessing definitions
from q5 import *

# Load the preprocessed data
df = pd.read_csv('fully_preprocessed_data.csv')

# Ensure 'rating_category' exists, if not, create it based on 'overall' rating
if 'rating_category' not in df.columns:
    df['rating_category'] = df['overall'].apply(lambda x: 'Good' if x >= 3 else 'Bad')

# Convert reviewTime to datetime format and extract year
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df['year'] = df['reviewTime'].dt.year

# Function to generate and save wordclouds for Good and Bad ratings
def generate_and_save_wordclouds(df):
    for rating_category in ['Good', 'Bad']:
        text = " ".join(review for review in df[df['rating_category'] == rating_category]['processed_reviewText'])
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{rating_category} Ratings Word Cloud")
        plt.savefig(f'{rating_category.lower()}_ratings_wordcloud.png')  # Save the wordcloud image
        plt.close()

generate_and_save_wordclouds(df)

# Function to plot and save a pie chart for Distribution of Ratings vs. No. of Reviews
def plot_and_save_pie_chart(df):
    ratings_distribution = df['overall'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(ratings_distribution, labels=ratings_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Ratings vs. Number of Reviews')
    plt.savefig('ratings_distribution_pie_chart.png')  # Save the pie chart
    plt.close()

plot_and_save_pie_chart(df)

# Function to report years with maximum reviews and highest number of customers
def report_years_with_max_reviews_and_customers(df):
    year_max_reviews = df['year'].value_counts().idxmax()
    year_max_customers = df.groupby('year')['reviewerID'].nunique().idxmax()
    
    report_content = f"Year with maximum reviews: {year_max_reviews}\nYear with the highest number of customers: {year_max_customers}\n"
    print(report_content)
    # Write to file, overwriting existing content
    with open('yearly_statistics.txt', 'w') as file:
        file.write(report_content)

report_years_with_max_reviews_and_customers(df)
