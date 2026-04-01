
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"7817_1.csv" 

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at {file_path}. Please check the path and filename.")

print("Original columns:", data.columns)

data = data.rename(columns={
    'reviews.username': 'user_id',
    'asins': 'item_id',
    'reviews.rating': 'rating'
})

data = data[['user_id', 'item_id', 'rating']].dropna()
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data = data.dropna(subset=['rating'])

sample_size = min(100000, len(data)) 
data = data.sample(sample_size, random_state=42)
print("Data shape after sampling:", data.shape)

item_user_matrix = data.pivot_table(index='item_id', columns='user_id', values='rating')
item_user_filled = item_user_matrix.fillna(0)

item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=item_user_matrix.index,
                                  columns=item_user_matrix.index)

def get_similar_items(item_id, n=5):
    if item_id not in item_similarity_df:
        return []
    return item_similarity_df[item_id].sort_values(ascending=False).iloc[1:n+1]

def recommend_items_item_based(user_id, n=5):
    if user_id not in item_user_matrix.columns:
        return []
    user_ratings = item_user_matrix[user_id].dropna()
    scores = {}
    for item, rating in user_ratings.items():
        for sim_item, sim_score in get_similar_items(item, n).items():
            if sim_item not in user_ratings:
                scores[sim_item] = scores.get(sim_item, 0) + sim_score * rating
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)

def recommend_items_user_based(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return []
    sim_users = user_similarity_df[user_id].sort_values(ascending=False)
    top_users = sim_users.iloc[1:6].index
    scores = {}
    for other in top_users:
        for item, rating in user_item_matrix.loc[other].items():
            if rating > 0 and item not in user_item_matrix.loc[user_id].dropna().index:
                scores[item] = scores.get(item, 0) + rating
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

train, test = train_test_split(data, test_size=0.2, random_state=42)
train_matrix = train.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
sim_train = cosine_similarity(train_matrix)

def calculate_rmse(sample_size=1000):
    preds, actual = [], []
    test_sample = test.sample(min(sample_size, len(test)), random_state=42)
    for _, row in test_sample.iterrows():
        u, i, r = row['user_id'], row['item_id'], row['rating']
        if u in train_matrix.index and i in train_matrix.columns:
            u_idx = train_matrix.index.get_loc(u)
            sim_scores = sim_train[u_idx]
            item_ratings = train_matrix[i]
            if np.sum(sim_scores) > 0:
                pred = np.dot(sim_scores, item_ratings) / np.sum(sim_scores)
                preds.append(pred)
                actual.append(r)
    return np.sqrt(mean_squared_error(actual, preds))

print("RMSE:", calculate_rmse())

def precision_at_k(user_id, recommender, k=5):
    recs = recommender(user_id, k)
    if not recs:
        return 0
    actual_items = set(item_user_matrix[user_id].dropna().index)
    recommended_items = [i[0] for i in recs]
    return sum(i in actual_items for i in recommended_items) / k

sample_user = item_user_matrix.columns[0]
print("Precision@K (Item-Based):", precision_at_k(sample_user, recommend_items_item_based))
print("Precision@K (User-Based):", precision_at_k(sample_user, recommend_items_user_based))

pop = data['item_id'].value_counts()
print("Top 10 Popular Items:\n", pop.head(10))
print("Top 10 Niche Items:\n", pop.tail(10))

plt.figure(figsize=(10,6))
sns.heatmap(item_similarity_df.iloc[:20,:20])
plt.title("Item Similarity Heatmap")
plt.show()

data = data.groupby('item_id').filter(lambda x: len(x) >= 10)
item_user_matrix = data.pivot_table(index='item_id', columns='user_id', values='rating')
item_user_filled = item_user_matrix.fillna(0)

item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

item_sample = item_user_matrix.index[0]
s = get_similar_items(item_sample, 10)

s = s[s > 0]
plt.figure(figsize=(10,6))
plt.barh(s.index, s.values, color='skyblue')
plt.xlabel("Similarity Score")
plt.title(f"Top Similar Items for {item_sample}")
plt.gca().invert_yaxis() 
plt.show()