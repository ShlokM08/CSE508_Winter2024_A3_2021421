
import pandas as pd

def parse_metadata(metadata_path, columns):
    
    chunk_size = 10000 
    for chunk in pd.read_json(metadata_path, lines=True, chunksize=chunk_size):
        for index, row in chunk.iterrows():
            yield row[columns].to_dict()

def preprocess_text(text):
  
    return text

def load_and_process_in_chunks(reviews_path, metadata_path, product_keyword):
    chunk_size = 10000  
    list_of_filtered_chunks = []  
    metadata_columns = ['asin', 'title', 'brand', 'category']  

    # Load and process metadata
    metadata_df = pd.DataFrame.from_records(parse_metadata(metadata_path, metadata_columns))

    # Load and process reviews in chunks
    for chunk in pd.read_json(reviews_path, lines=True, chunksize=chunk_size):
        # Assuming 'reviewText' column exists and applying preprocessing on it
        chunk['processed_reviewText'] = chunk['reviewText'].astype(str).apply(preprocess_text)
        # Filter based on the presence of the product keyword in the processed review text
        filtered_chunk = chunk[chunk['processed_reviewText'].str.contains(product_keyword, case=False)]
        # If no rows match the filter, skip to the next chunk
        if filtered_chunk.empty:
            continue
        # Merge with metadata to get additional product details
        filtered_chunk = pd.merge(filtered_chunk, metadata_df, on='asin', how='left')
        # Append the filtered and merged chunk to the list
        list_of_filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    return pd.concat(list_of_filtered_chunks, ignore_index=True) if list_of_filtered_chunks else pd.DataFrame()

# Set file paths
reviews_path = 'C:\\Users\\Shlok Mehroliya\\Downloads\\IR_assignment3\\Electronics_5.json.gz'
metadata_path = 'C:\\Users\\Shlok Mehroliya\\Downloads\\IR_assignment3\\meta_Electronics.json.gz'


# Example usage
processed_data = load_and_process_in_chunks(reviews_path, metadata_path, 'Headphone')
print(processed_data.head())
# Save the processed data to a Parquet file
# After processing
processed_data.to_csv('processed_data.csv', index=False)


    
 