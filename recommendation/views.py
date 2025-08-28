from django.shortcuts import render
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


from collections import UserDict

class BracketDict(UserDict):
    def __getitem__(self, key):
        return super().data[key]

# Load dataset
df = pd.read_csv("TeluguMovies_dataset.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Movie", "PosterURL"])

# Fill missing values
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Ensure correct types
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Rating"].fillna(df["Rating"].mean(), inplace=True)
df["Genre"] = df["Genre"].astype(str)
df["Movie_lower"] = df["Movie"].str.lower()

# Combine features
df["combined_features"] = (
    df["Genre"].astype(str) + " " +
    df["Overview"].astype(str) + " " +
    df["Certificate"].astype(str) + " " +
    df["Language"].astype(str)
)

# TF-IDF + KMeans
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

n_clusters = 20  # you can tune this
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(tfidf_matrix)


def recommend_movies(query, top_n=15):
    query = query.strip()

    # Exact match
    exact_matches = df[df["Movie"].str.lower() == query.lower()]
    if not exact_matches.empty:
        searched_index = exact_matches.index[0]
    else:
        # Fuzzy match
        match, score = process.extractOne(query, df["Movie"].tolist())
        if score < 40:
            return pd.DataFrame()
        searched_index = df[df["Movie"] == match].index[0]

    # Get cluster of searched movie
    searched_cluster = df.loc[searched_index, "Cluster"]

    # Get all movies in same cluster
    cluster_movies = df[df["Cluster"] == searched_cluster]

    # Exclude searched movie, keep top N
    recommendations = cluster_movies[cluster_movies.index != searched_index].head(top_n - 1)

    # Final list: searched movie + recommendations
    final_df = pd.concat([df.loc[[searched_index]], recommendations])

    return final_df


from django.utils.safestring import mark_safe
import json

def home(request):
    results = []
    searched_movie_name = ""
    if request.method == "POST":
        query = request.POST.get("query")
        if query:
            df_results = recommend_movies(query)
            if not df_results.empty:
                # Convert dataframe rows into JSON-like strings
                results = [
                    {k: v for k, v in row.items()} for row in df_results.to_dict(orient="records")
                ]
                searched_movie_name = df_results.iloc[0]["Movie"]

    # Pass results as JSON string that template can parse directly
    return render(request, "index.html", {
        "results": results,
        "searched_movie_name": searched_movie_name
    })


