from pathlib import Path
import ast
import pickle

import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MOVIES_PATH = MODEL_DIR / "movies.pkl"
SIMILARITY_PATH = MODEL_DIR / "similarity.pkl"


def _convert(obj):
    return [item["name"] for item in ast.literal_eval(obj)]


def _convert3(obj):
    names = []
    counter = 0
    for item in ast.literal_eval(obj):
        if counter == 3:
            break
        names.append(item["name"])
        counter += 1
    return names


def _fetch_director(obj):
    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            return [item["name"]]
    return []


def _stem(text):
    ps = PorterStemmer()
    return " ".join(ps.stem(word) for word in text.split())


# ---------- Build Model ----------
def build_model():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies_df = movies_df.merge(credits, on="title")
    movies_df = movies_df[["id", "title", "cast", "crew", "overview", "keywords", "genres"]].copy()
    movies_df = movies_df.dropna().copy()

    movies_df["genres"] = movies_df["genres"].apply(_convert)
    movies_df["keywords"] = movies_df["keywords"].apply(_convert)
    movies_df["cast"] = movies_df["cast"].apply(_convert3)
    movies_df["crew"] = movies_df["crew"].apply(_fetch_director)
    movies_df["overview"] = movies_df["overview"].apply(lambda x: x.split())

    movies_df["genres"] = movies_df["genres"].apply(lambda x: [item.replace(" ", "") for item in x])
    movies_df["keywords"] = movies_df["keywords"].apply(lambda x: [item.replace(" ", "") for item in x])
    movies_df["cast"] = movies_df["cast"].apply(lambda x: [item.replace(" ", "") for item in x])
    movies_df["crew"] = movies_df["crew"].apply(lambda x: [item.replace(" ", "") for item in x])

    movies_df["tags"] = movies_df["overview"] + movies_df["genres"] + movies_df["keywords"] + movies_df["cast"] + movies_df["crew"]
    new_df = movies_df[["id", "title", "tags"]].copy()
    new_df = new_df.rename(columns={"id": "movie_id"})
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x)).str.lower()
    new_df["tags"] = new_df["tags"].apply(_stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    with MOVIES_PATH.open("wb") as f:
        pickle.dump(new_df[["movie_id", "title"]], f)
    with SIMILARITY_PATH.open("wb") as f:
        pickle.dump(similarity, f)

    return new_df[["movie_id", "title"]], similarity


# ---------- Load Model ----------
@st.cache_data
def load_data():
    if not MOVIES_PATH.exists() or not SIMILARITY_PATH.exists():
        return build_model()

    with MOVIES_PATH.open("rb") as f:
        movies = pickle.load(f)
    with SIMILARITY_PATH.open("rb") as f:
        similarity = pickle.load(f)

    if "movie_id" not in movies.columns and "id" in movies.columns:
        movies = movies.rename(columns={"id": "movie_id"})
    return movies, similarity


movies, similarity = load_data()

# ---------- TMDB Poster Fetch ----------
API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        resp = requests.get(url, timeout=5).json()
        path = resp.get('poster_path')
        if path:
            return "https://image.tmdb.org/t/p/w500/" + path
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# ---------- Recommend ----------
def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    names, posters = [], []
    for i, _ in scores:
        movie_id = movies.iloc[i].movie_id
        names.append(movies.iloc[i].title)
        posters.append(fetch_poster(movie_id))
    return names, posters

# ---------- UI ----------
st.title("🎬 Movie Recommendation System")
st.write("Pick a movie and get 5 similar recommendations!")

movie_list = movies['title'].values
selected = st.selectbox("Choose a movie:", movie_list)

if st.button("🎯 Recommend"):
    with st.spinner("Fetching recommendations..."):
        names, posters = recommend(selected)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.image(poster, use_container_width=True)
            st.caption(name)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Content-based filtering using cosine similarity.")
    st.write(f"📚 Total movies: **{len(movies)}**")