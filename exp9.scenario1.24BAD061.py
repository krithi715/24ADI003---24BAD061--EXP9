import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
ratings = pd.read_csv('u.data', sep='\t', header=None,
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

movies = pd.read_csv('u.item', sep='|', encoding='latin-1',
                     header=None, usecols=[0, 1],
                     names=['movie_id', 'title'])

data = pd.merge(ratings, movies, on='movie_id')

print("\nSample Data:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

user_item_matrix = data.pivot_table(index='user_id',
                                    columns='title',
                                    values='rating')

user_item_filled = user_item_matrix.fillna(0)

user_similarity = cosine_similarity(user_item_filled)

user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_item_matrix.index,
                                 columns=user_item_matrix.index)

def get_similar_users(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    return similar_users.iloc[1:n+1]

def predict_ratings(user_id, top_n=5):
    similar_users = get_similar_users(user_id, top_n)
    
    user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index
    
    predictions = {}
    
    for movie in unseen_movies:
        weighted_sum = 0
        similarity_sum = 0
        
        for sim_user, similarity in similar_users.items():
            rating = user_item_matrix.loc[sim_user, movie]
            
            if not np.isnan(rating):
                weighted_sum += similarity * rating
                similarity_sum += similarity
        
        if similarity_sum != 0:
            predictions[movie] = weighted_sum / similarity_sum
    
    return predictions

def recommend_movies(user_id, n=5):
    predictions = predict_ratings(user_id)
    
    recommended = sorted(predictions.items(),
                         key=lambda x: x[1],
                         reverse=True)
    
    return recommended[:n]


print("\nTop Recommendations for User 1:")
for movie, rating in recommend_movies(1, 5):
    print(movie, "->", round(rating, 2))

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = train.pivot_table(index='user_id',
                                 columns='movie_id',
                                 values='rating').fillna(0)

similarity = cosine_similarity(train_matrix)


def evaluate():
    preds = []
    actuals = []
    
    for _, row in test.iterrows():
        user = row['user_id']
        movie = row['movie_id']
        actual = row['rating']
        
        if user in train_matrix.index and movie in train_matrix.columns:
            user_idx = train_matrix.index.get_loc(user)
            sim_scores = similarity[user_idx]
            movie_ratings = train_matrix[movie]
            
            if np.sum(sim_scores) != 0:
                pred = np.dot(sim_scores, movie_ratings) / np.sum(sim_scores)
                
                preds.append(pred)
                actuals.append(actual)
    
    return preds, actuals


preds, actuals = evaluate()

rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)

print("\nEvaluation Metrics:")
print("RMSE:", rmse)
print("MAE:", mae)

sparsity = 1 - (np.count_nonzero(user_item_matrix) / user_item_matrix.size)
print("\nMatrix Sparsity:", sparsity)
plt.figure(figsize=(16,8))
sns.heatmap(user_item_filled.iloc[:20, :15])

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0)

plt.title("User-Item Matrix Heatmap")

plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(user_similarity_df.iloc[:20, :20])
plt.title("User Similarity Matrix")
plt.show()

recommended = recommend_movies(1, 10)

if recommended:
    movies_list, scores = zip(*recommended)

    plt.figure(figsize=(8,5))
    plt.barh(movies_list, scores)
    plt.title("Top Recommended Movies")
    plt.xlabel("Predicted Rating")
    plt.gca().invert_yaxis()
    plt.show()
