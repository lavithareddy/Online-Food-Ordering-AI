from flask import Flask, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# ---------- LOAD DATA ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "dataset", "food_data.csv")

data = pd.read_csv(data_path)

# Encode categorical columns .
category_encoder = LabelEncoder()
cuisine_encoder = LabelEncoder()

data["category_code"] = category_encoder.fit_transform(data["category"])
data["cuisine_code"] = cuisine_encoder.fit_transform(data["cuisine"])

# Feature matrix 
features = data[["category_code", "cuisine_code", "price", "rating"]]
similarity = cosine_similarity(features)

# ---------- RECOMMEND FUNCTION ----------
def recommend_food(food_name):
    food_name = food_name.strip().lower()
    data["food_name_clean"] = data["food_name"].str.lower()
    if food_name not in data["food_name_clean"].values:
        return []

    index = data[data["food_name_clean"] == food_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

recommendations = []
for i in scores:
    food = data.iloc[i[0]]["food_name"]
    if food.lower() != food_name.lower():
        recommendations.append(food)
    if len(recommendations) == 3:
        break

    return recommendations
# ---------- ROUTES ----------
@app.route("/")
def home():
    return "AI Food Recommendation API is running"
@app.route("/recommend/<food_name>")
def recommend(food_name):
    result = recommend_food(food_name)
    if result:
        return jsonify({
            "selected_food": food_name,
            "recommendations": result
        })
    else:
        return jsonify({"error": "Food item not found"})

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=False)
