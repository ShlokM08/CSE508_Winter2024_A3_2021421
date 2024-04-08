import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

class CollaborativeFiltering:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.normalize_data()

    def normalize_data(self):
    # Use pivot_table with aggregation function, here we use 'mean' to handle duplicate entries
        pivot_table = self.data_frame.pivot_table(index='reviewerID', columns='asin', values='overall', fill_value=0, aggfunc='mean')
        scaler = MinMaxScaler()
        self.user_item_matrix = pd.DataFrame(scaler.fit_transform(pivot_table), index=pivot_table.index, columns=pivot_table.columns)
        self.item_user_matrix = self.user_item_matrix.transpose()


    @staticmethod
    def calculate_cosine_similarity(matrix):
        return cosine_similarity(matrix)

    @staticmethod
    def find_nearest_neighbors(similarity_matrix, number_of_neighbors):
        return np.argsort(-similarity_matrix, axis=1)[:, :number_of_neighbors]

    @staticmethod
    def mean_absolute_error_nonzero(actual, predicted):
        nonzero_indices = np.nonzero(actual)
        actual_nonzero = actual[nonzero_indices]
        predicted_nonzero = predicted[nonzero_indices]
        return mean_absolute_error(actual_nonzero, predicted_nonzero)

    def predict_ratings(self, similarity_matrix, nearest_neighbors_indices, scaled_matrix):
        predictions = np.zeros(scaled_matrix.shape)
        for user_index in range(scaled_matrix.shape[0]):
            for neighbor_index in nearest_neighbors_indices[user_index]:
                predictions[user_index, :] += similarity_matrix[user_index, neighbor_index] * scaled_matrix.iloc[neighbor_index, :]
            sum_similarity = np.sum(np.abs(similarity_matrix[user_index, nearest_neighbors_indices[user_index]]))
            if sum_similarity > 0:
                predictions[user_index] /= sum_similarity
        return predictions

    def collaborative_filtering(self, scaled_matrix, neighbors_list=[10, 20, 30, 40, 50]):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mean_absolute_errors = {}
        for neighbors in neighbors_list:
            mae_list = []
            for train_index, test_index in kf.split(scaled_matrix):
                train_matrix, test_matrix = scaled_matrix.iloc[train_index], scaled_matrix.iloc[test_index]
                similarity_matrix = self.calculate_cosine_similarity(train_matrix.to_numpy())
                nearest_neighbors_indices = self.find_nearest_neighbors(similarity_matrix, neighbors)
                predicted_matrix = self.predict_ratings(similarity_matrix, nearest_neighbors_indices, train_matrix)
                actual_matrix = test_matrix.to_numpy()
                mae = self.mean_absolute_error_nonzero(actual_matrix, predicted_matrix)
                mae_list.append(mae)
            mean_absolute_errors[neighbors] = np.mean(mae_list)
        return mean_absolute_errors

np.random.seed(42)
sample_data = pd.DataFrame({
    'reviewerID': np.random.randint(1, 100, 1000),
    'asin': np.random.randint(1, 20, 1000),
    'overall': np.random.randint(1, 6, 1000)
})

cf_instance = CollaborativeFiltering(sample_data)
user_user_mae_values = cf_instance.collaborative_filtering(cf_instance.user_item_matrix)
item_item_mae_values = cf_instance.collaborative_filtering(cf_instance.item_user_matrix)

# Save the MAE values to a file in the current directory
mae_values_filename = "mae_values.txt"  # You can also specify an absolute path
with open(mae_values_filename, 'w') as file:
    file.write("User-User Collaborative Filtering MAE values:\n")
    file.write(str(user_user_mae_values))
    file.write("\n\nItem-Item Collaborative Filtering MAE values:\n")
    file.write(str(item_item_mae_values))

# Visualize and save the plot in the current directory
plot_filename = "cf_mae_comparison.png"  # You can also specify an absolute path
plt.figure(figsize=(10, 5))
plt.plot(list(user_user_mae_values.keys()), list(user_user_mae_values.values()), label='User-User CF', marker='o')
plt.plot(list(item_item_mae_values.keys()), list(item_item_mae_values.values()), label='Item-Item CF', marker='x')
plt.xlabel('Number of Nearest Neighbors')
plt.ylabel('Mean Absolute Error')
plt.title('Collaborative Filtering MAE Comparison')
plt.legend()
plt.grid(True)
plt.savefig(plot_filename)

pSum = sample_data.groupby('asin')['overall'].sum().sort_values(ascending=False)

# Select the top 10 products with the highest sum of ratings
ans = pSum.head(10)

# Print the top 10 products
print("Top 10 Products by User Sum Ratings:")
print(ans)