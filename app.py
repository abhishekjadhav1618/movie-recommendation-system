import os
import base64
import requests
import streamlit as st
import pandas as pd
from recommendation import load_data, recommend

# Set up page configurations
st.set_page_config(
    page_title="CineMatch - AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Base64 image converter for local offline fallback poster
def get_base64_image(image_path):
    """Converts a local image file to base64 string for HTML embedding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[Error] Failed to load local asset base64: {e}")
        return ""

# Fetch movie poster path from TMDB API with caching
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, api_key):
    """
    Queries TMDB API to fetch movie poster URL.
    Returns None if api call fails or rate-limited.
    """
    if not api_key:
        return None
        
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        # 1.5 seconds timeout to ensure responsive UI
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        # Ignore network errors and timeout gracefully
        pass
    return None

# Load custom CSS styles for premium glassmorphism dark-theme
st.markdown("""
<style>
    /* Sleek Dark Mode Background & Styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Title Gradient styling */
    .title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 1.5rem;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0) 100%);
        border-radius: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, #F43F5E, #8B5CF6, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
    }
    
    /* Dropdown UI styling */
    .stSelectbox label {
        color: #E2E8F0 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    
    /* Button custom hover animations */
    .stButton button {
        background: linear-gradient(90deg, #F43F5E, #8B5CF6) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.6rem 2.5rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(244, 63, 94, 0.3) !important;
        width: 100% !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
        border: none !important;
    }
    
    /* Movie Card Styles with Hover Effects */
    .movie-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .movie-card:hover {
        transform: translateY(-8px);
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.25);
    }
    .movie-poster-container {
        position: relative;
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 12px;
        aspect-ratio: 2/3;
    }
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    .movie-card:hover .movie-poster {
        transform: scale(1.06);
    }
    .movie-info {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        justify-content: space-between;
        padding: 4px 2px;
    }
    .movie-title {
        color: #F8FAFC;
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.3;
        margin-bottom: 8px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .movie-meta-label {
        font-size: 0.7rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 0.5px;
        margin-bottom: 1px;
    }
    .movie-meta-val {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-bottom: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-genre-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.25);
        color: #A78BFA;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 4px;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# Build Header
st.markdown("""
<div class="title-container">
    <div class="main-title">🎬 CineMatch</div>
    <div class="sub-title">Content-Based Movie Recommendation System using AI/ML</div>
</div>
""", unsafe_allow_html=True)

# Sidebar setup for TMDB API configurations & explanations
st.sidebar.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60", width=300)
st.sidebar.markdown("### ⚙️ Settings")

# Fallback developer TMDB API Key (Standard free tier key used in ML demos)
fallback_key = "8265bd1679663a7ea12ac168da84d2e8"
user_api_key = st.sidebar.text_input(
    "TMDB API Key (v3 auth)", 
    value=fallback_key, 
    type="password",
    help="Provided standard key is preconfigured. If you have your own TMDB API key, feel free to use it."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧠 How It Works
1. **Metadata Aggregation:** We load genres, keywords, cast, director, and plot overview for each movie.
2. **Space Collapsing:** Spaces in cast, director, and genre names are removed (e.g. *Christopher Nolan* $\rightarrow$ *ChristopherNolan*) so they are processed as single tokens.
3. **Porter Stemming:** Standard vocabulary normalization using NLTK to reduce words to their stem form.
4. **Vectorization:** Convert processed tags into a 5,000-dimensional numerical representation via Bag-of-Words.
5. **Cosine Similarity:** Compute similarity scores using angular distance between vectors to fetch recommendations.
""")

# Load local offline fallback poster
base_dir = os.path.dirname(os.path.abspath(__file__))
no_poster_path = os.path.join(base_dir, "assets", "no_poster.png")
no_poster_base64 = get_base64_image(no_poster_path)
no_poster_src = f"data:image/png;base64,{no_poster_base64}" if no_poster_base64 else ""

# Load dataset
try:
    movies_df, similarity_matrix = load_data()
    movie_list = movies_df['title'].values
    
    # ------------------ FRONTEND SEARCH SECTION ------------------
    st.markdown("### Find Your Next Favorite Movie")
    
    # Dropdown selector
    selected_movie = st.selectbox(
        "Type or select a movie from the catalog:",
        movie_list,
        index=movie_list.tolist().index("Avatar") if "Avatar" in movie_list else 0
    )
    
    # Recommend Button
    if st.button("🔍 Recommend Similar Movies"):
        with st.spinner("Analyzing cinematic features..."):
            recommendations = recommend(selected_movie, movies_df, similarity_matrix)
            
            if not recommendations:
                st.error("No recommendations found or an error occurred.")
            else:
                st.success(f"Here are the top 10 movies similar to **{selected_movie}**:")
                
                # Split display into 2 rows of 5 columns each
                row1 = st.columns(5)
                row2 = st.columns(5)
                
                for idx, movie in enumerate(recommendations):
                    # Fetch Poster from API
                    poster_url = fetch_poster(movie['movie_id'], user_api_key)
                    poster_src = poster_url if poster_url else no_poster_src
                    
                    # Shorten long fields for display integrity
                    genres = movie['genres']
                    genre_display = genres.split(",")[0] if "," in genres else genres
                    
                    # Generate HTML Card content
                    card_html = f"""
                    <div class="movie-card">
                        <div class="movie-poster-container">
                            <img class="movie-poster" src="{poster_src}" alt="{movie['title']} Poster">
                        </div>
                        <div class="movie-info">
                            <div class="movie-title" title="{movie['title']}">{movie['title']}</div>
                            <div>
                                <div class="movie-meta-label">Director</div>
                                <div class="movie-meta-val" title="{movie['director']}">{movie['director'] if movie['director'] and movie['director'] != 'nan' else 'Unknown'}</div>
                                <div class="movie-meta-label">Primary Genre</div>
                                <div class="movie-genre-badge" title="{movie['genres']}">{genre_display if genre_display and genre_display != 'nan' else 'Drama'}</div>
                            </div>
                        </div>
                    </div>
                    """
                    
                    # Select corresponding column layout
                    if idx < 5:
                        with row1[idx]:
                            st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        with row2[idx - 5]:
                            st.markdown(card_html, unsafe_allow_html=True)
                            
except FileNotFoundError as e:
    st.warning("⚠️ Ready to build models!")
    st.info(
        "Welcome! It looks like you haven't processed the dataset yet. "
        "Please run `python preprocess.py` in your terminal to download the dataset "
        "and generate the similarity matrix."
    )
except Exception as e:
    st.error(f"Failed to load application: {e}")
