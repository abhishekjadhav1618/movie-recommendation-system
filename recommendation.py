import os
import pickle
import pandas as pd

def load_data():
    """
    Loads the preprocessed movies dataframe and similarity matrix.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, "movies.pkl")
    similarity_path = os.path.join(base_dir, "similarity.pkl")
    
    # Check if files exist
    if not os.path.exists(movies_path) or not os.path.exists(similarity_path):
        raise FileNotFoundError(
            "Preprocessed files movies.pkl or similarity.pkl not found! "
            "Please run preprocess.py first to generate the models."
        )
        
    with open(movies_path, 'rb') as f:
        movies_df = pickle.load(f)
        
    with open(similarity_path, 'rb') as f:
        similarity_matrix = pickle.load(f)
        
    return movies_df, similarity_matrix

def recommend(movie_title, movies_df, similarity_matrix):
    """
    Suggests 10 similar movies based on the input movie title.
    
    Parameters:
        movie_title (str): The exact title of the selected movie.
        movies_df (pd.DataFrame): The preprocessed movies DataFrame.
        similarity_matrix (np.ndarray): The cosine similarity matrix.
        
    Returns:
        list of dicts: A list containing recommendations with metadata (id, title, genres, director).
    """
    try:
        # Find index of the selected movie
        movie_index = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index[0]
        
        # Get similarity scores for this movie
        distances = similarity_matrix[movie_index]
        
        # Sort and select top 10 similar movies (excluding the movie itself)
        # Each element in sorted list is a tuple: (index, similarity_score)
        similar_movies_raw = sorted(
            list(enumerate(distances)), 
            reverse=True, 
            key=lambda x: x[1]
        )[1:11]
        
        recommendations = []
        for index, score in similar_movies_raw:
            movie_row = movies_df.iloc[index]
            
            # Format genres and director list nicely for display
            genres_list = movie_row['genres']
            director_list = movie_row['director']
            
            genres_display = ", ".join(genres_list) if isinstance(genres_list, list) else str(genres_list)
            director_display = ", ".join(director_list) if isinstance(director_list, list) else str(director_list)
            
            recommendations.append({
                "movie_id": int(movie_row['movie_id']),
                "title": movie_row['title'],
                "genres": genres_display,
                "director": director_display,
                "score": float(score)
            })
            
        return recommendations
        
    except IndexError:
        print(f"[Error] Movie '{movie_title}' not found in the dataset.")
        return []
    except Exception as e:
        print(f"[Error] An error occurred in the recommendation engine: {e}")
        return []
