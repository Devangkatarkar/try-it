import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
from google.colab import files  # For file upload

def load_data(filepath_or_buffer):  # Modified to handle file uploads
    try:
        df = pd.read_csv(filepath_or_buffer)
        df.fillna("", inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath_or_buffer}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ... (rest of the functions: create_tfidf_matrix, get_recommendations remain the same)

# File Upload in Colab
uploaded = files.upload() #This will open a file selection dialog in your browser

if uploaded:
    filename = list(uploaded.keys())[0] # Get the filename
    anime_data = load_data(io.StringIO(uploaded[filename].decode('utf-8'))) # Load directly from uploaded data

    if anime_data is not None:
        anime_data['combined_features'] = anime_data['Genre'] + " " + anime_data['Synopsis']

        tfidf_matrix = create_tfidf_matrix(anime_data, 'combined_features')

        anime_title = input("Enter an anime title: ")
        recommended_anime = get_recommendations(anime_title, anime_data, tfidf_matrix)

        if recommended_anime:
            print("Recommended anime:")
            for anime in recommended_anime:
                print(f"- {anime}")

else:
    print("No file uploaded.")
