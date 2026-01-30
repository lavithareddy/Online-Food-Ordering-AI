import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Get base directory (Online-Food-Ordering-AI)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build dataset path safely
data_path = os.path.join(BASE_DIR, "dataset", "food_data.csv")

# Load dataset
data = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(data.head())

# Encode text columns
category_encoder = LabelEncoder()
cuisine_encoder = LabelEncoder()

data["category_code"] = category_encoder.fit_transform(data["category"])
data["cuisine_code"] = cuisine_encoder.fit_transform(data["cuisine"])

# Feature matrix
features = data[["category_code", "cuisine_code", "price", "rating"]]

# Cosine similarity
similarity = cosine_similarity(features)

# Recommendation function
def recommend_food(food_name):
    index = data[data["food_name"] == food_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:4]:
        recommendations.append(data.iloc[i[0]]["food_name"])

    return recommendations

# Test
print("\nBecause you liked Chicken Biryani, we recommend:")
print(recommend_food("Chicken Biryani"))
