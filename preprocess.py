import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK packages are downloaded (no heavy downloads needed for PorterStemmer)
# nltk.download() is not required for PorterStemmer as it is built-in, but importing it is enough.

def download_file(url, filepath):
    """
    Downloads a file from a URL to a local filepath with a progress indicator.
    """
    if os.path.exists(filepath):
        print(f"[Info] File already exists at: {filepath}")
        return
    
    print(f"[Info] Downloading from {url} to {filepath}...")
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Simple download progress display
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
            print(f"\rDownloading... {percent:.2f}% ({downloaded / (1024*1024):.2f} MB)", end="")
            
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print("\n[Success] Download complete.")
    except Exception as e:
        print(f"\n[Error] Failed to download {url}: {e}")
        raise e

# Helpers to extract text fields from TMDB dataset JSON strings
def convert_genres_keywords(obj):
    """Extract list of names from string representation of JSON list of dicts."""
    try:
        names = []
        for item in ast.literal_eval(obj):
            names.append(item['name'])
        return names
    except (ValueError, SyntaxError):
        return []

def convert_cast(obj):
    """Extract top 3 cast actor names from JSON string representation."""
    try:
        names = []
        counter = 0
        for item in ast.literal_eval(obj):
            if counter < 3:
                names.append(item['name'])
                counter += 1
            else:
                break
        return names
    except (ValueError, SyntaxError):
        return []

def fetch_director(obj):
    """Extract the director name from JSON string representation of crew."""
    try:
        for item in ast.literal_eval(obj):
            if item.get('job') == 'Director':
                return [item['name']]
        return []
    except (ValueError, SyntaxError):
        return []

def collapse_spaces(L):
    """Removes spaces between words to create single tokens (e.g. 'Science Fiction' -> 'ScienceFiction')."""
    return [name.replace(" ", "") for name in L]

def run_pipeline():
    print("=== Starting Movie Recommendation System Data Pipeline ===")
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Dataset source URLs (Hugging Face verified mirrors)
    movies_url = "https://huggingface.co/datasets/Adpacci/tmdb-5000/resolve/main/tmdb_5000_movies.csv"
    credits_url = "https://huggingface.co/datasets/Adpacci/tmdb-5000/resolve/main/tmdb_5000_credits.csv"
    
    raw_movies_path = os.path.join(dataset_dir, "tmdb_5000_movies.csv")
    raw_credits_path = os.path.join(dataset_dir, "tmdb_5000_credits.csv")
    
    # Download files
    download_file(movies_url, raw_movies_path)
    download_file(credits_url, raw_credits_path)
    
    # 1. Load Movie Dataset
    print("[1/8] Loading raw movies and credits datasets...")
    movies_df = pd.read_csv(raw_movies_path)
    credits_df = pd.read_csv(raw_credits_path)
    
    # Merge datasets on Title
    print("[2/8] Merging movies and credits dataframes...")
    merged_df = movies_df.merge(credits_df, on='title')
    
    # 2. Clean Missing Values & Columns Selection
    # Keep useful columns: id, title, overview, genres, keywords, cast, crew
    # Note: id corresponds to tmdb_id which is needed for fetching posters from API
    print("[3/8] Selecting features and cleaning missing values...")
    cleaned_df = merged_df[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
    cleaned_df.dropna(inplace=True)
    
    # 3. Select useful features & Parse JSON Columns
    print("[4/8] Parsing JSON strings for features...")
    cleaned_df['genres'] = cleaned_df['genres'].apply(convert_genres_keywords)
    cleaned_df['keywords'] = cleaned_df['keywords'].apply(convert_genres_keywords)
    cleaned_df['cast'] = cleaned_df['cast'].apply(convert_cast)
    cleaned_df['director'] = cleaned_df['crew'].apply(fetch_director)
    
    # Clean spacing for content representation (treating entities as unique tokens)
    cleaned_df['genres'] = cleaned_df['genres'].apply(collapse_spaces)
    cleaned_df['keywords'] = cleaned_df['keywords'].apply(collapse_spaces)
    cleaned_df['cast'] = cleaned_df['cast'].apply(collapse_spaces)
    cleaned_df['director'] = cleaned_df['director'].apply(collapse_spaces)
    
    # Split overview text into a list of words
    cleaned_df['overview'] = cleaned_df['overview'].apply(lambda x: x.split())
    
    # 4. Merge important features into 'tags'
    print("[5/8] Creating unified 'tags' feature...")
    cleaned_df['tags'] = (
        cleaned_df['overview'] + 
        cleaned_df['genres'] + 
        cleaned_df['keywords'] + 
        cleaned_df['cast'] + 
        cleaned_df['director']
    )
    
    # Convert lists back to text strings
    processed_df = cleaned_df[['id', 'title', 'genres', 'director', 'tags']].copy()
    processed_df['tags'] = processed_df['tags'].apply(lambda x: " ".join(x))
    
    # Apply lowercasing
    processed_df['tags'] = processed_df['tags'].apply(lambda x: x.lower())
    
    # Apply PorterStemmer for vocabulary normalization (stemming)
    print("[6/8] Performing text stemming...")
    ps = PorterStemmer()
    def stem_text(text):
        return " ".join([ps.stem(word) for word in text.split()])
        
    processed_df['tags'] = processed_df['tags'].apply(stem_text)
    
    # 5. Vectorize using CountVectorizer
    print("[7/8] Vectorizing text tags using CountVectorizer...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(processed_df['tags']).toarray()
    
    # 6. Calculate Cosine Similarity
    print("[8/8] Calculating Cosine Similarity Matrix...")
    similarity = cosine_similarity(vectors)
    
    # Cast to float32 to reduce pickle file size (saves ~50% space with zero accuracy loss)
    similarity = similarity.astype(np.float32)
    
    # Save the processed dataset to dataset/movies.csv
    cleaned_csv_path = os.path.join(dataset_dir, "movies.csv")
    
    # Let's save a clean representation for the CSV file
    # We join list columns back to string representations
    csv_export_df = cleaned_df[['id', 'title', 'genres', 'director', 'overview']].copy()
    csv_export_df['genres'] = csv_export_df['genres'].apply(lambda x: ", ".join(x))
    csv_export_df['director'] = csv_export_df['director'].apply(lambda x: ", ".join(x))
    csv_export_df['overview'] = csv_export_df['overview'].apply(lambda x: " ".join(x))
    csv_export_df.rename(columns={'id': 'movie_id'}, inplace=True)
    
    # Also add the processed 'tags' column to the CSV for transparency
    csv_export_df['tags'] = processed_df['tags']
    csv_export_df.to_csv(cleaned_csv_path, index=False)
    print(f"[Success] Saved processed dataset to: {cleaned_csv_path}")
    
    # 8. Save similarity matrix and processed data to Pickle files
    movies_pkl_path = os.path.join(base_dir, "movies.pkl")
    similarity_pkl_path = os.path.join(base_dir, "similarity.pkl")
    
    # Save movies list (we only need movie_id, title, genres, director, and tags in the app)
    app_movies_df = processed_df[['id', 'title', 'genres', 'director', 'tags']].rename(columns={'id': 'movie_id'})
    with open(movies_pkl_path, 'wb') as f:
        pickle.dump(app_movies_df, f)
    print(f"[Success] Saved movies pickle to: {movies_pkl_path}")
    
    with open(similarity_pkl_path, 'wb') as f:
        pickle.dump(similarity, f)
    print(f"[Success] Saved similarity matrix pickle to: {similarity_pkl_path}")
    
    # Clean up the large raw CSV files to keep repository slim
    print("[Info] Cleaning up raw heavy datasets...")
    try:
        os.remove(raw_movies_path)
        os.remove(raw_credits_path)
        print("[Success] Cleanup completed.")
    except Exception as e:
        print(f"[Warning] Failed to delete raw files during cleanup: {e}")
        
    print("=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline()
