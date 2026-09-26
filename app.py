import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ---------------- Load CSVs & Build Model ----------------
@st.cache_data(show_spinner="Building recommendation model...")
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    movies = movies.merge(credits, on='title')
    if 'poster_path' not in movies.columns:
        movies['poster_path'] = None
    movies = movies[['movie_id', 'title', 'overview', 'genres',
                     'keywords', 'cast', 'crew', 'poster_path']]
    movies.dropna(subset=['overview'], inplace=True)
    movies.reset_index(drop=True, inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def convert3(text):
        return [i['name'] for i in ast.literal_eval(text)][:3]

    def fetch_director(text):
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

    movies['genres']   = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast']     = movies['cast'].apply(convert3)
    movies['crew']     = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = (movies['overview'] + movies['genres'] +
                      movies['keywords'] + movies['cast'] + movies['crew'])
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies[['movie_id', 'title', 'poster_path']], similarity

movies, similarity = load_data()

# ---------------- Poster from local CSV (NO API KEY NEEDED) ----------------
def fetch_poster(poster_path):
    if pd.isna(poster_path) or not poster_path:
        return "https://via.placeholder.com/500x750?text=No+Poster"
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

# ---------------- Recommend ----------------
def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    scores = sorted(list(enumerate(similarity[idx])),
                    key=lambda x: x[1], reverse=True)[1:6]

    names, posters = [], []
    for i, _ in scores:
        row = movies.iloc[i]
        names.append(row['title'])
        posters.append(fetch_poster(row['poster_path']))
    return names, posters

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Pick a movie:", movies['title'].values)

if st.button("Show Recommendations"):
    with st.spinner("Finding similar movies..."):
        names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.image(poster, use_container_width=True)
            st.caption(name)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Content-based filtering using cosine similarity.")
    st.write(f"📚 Total movies: **{len(movies)}**")