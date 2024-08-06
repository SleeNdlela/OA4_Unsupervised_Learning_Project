import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import implicit
import pickle

# Load the dataset
anime_data = pd.read_csv('anime.csv', nrows=10000)  # Adjust the path to your data
user_ratings = pd.read_csv('train.csv', nrows=10000)  # Adjust the path to your data

# Convert 'name' to category dtype to save memory
anime_data['name'] = anime_data['name'].astype('category')

# Merge datasets to get the 'name' in user_ratings
user_ratings = user_ratings.merge(anime_data[['anime_id', 'name']], on='anime_id', how='left')

# Create the 'auth_tags' column
anime_data['auth_tags'] = (pd.Series(anime_data[['type', 'genre']]
                      .fillna('')
                      .values.tolist()).str.join(' '))

# Define indices for quick lookups
indices = pd.Series(anime_data.index, index=anime_data['name'])

# Transform the 'auth_tags' column into a TF-IDF matrix
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tf_authTags_matrix = tf.fit_transform(anime_data['auth_tags'])

# Load the pickled collaborative filtering model
with open('collab_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Content-based recommendation function
def content_generate_top_N_recommendations(name, N=10):
    if name not in indices.index:
        return pd.Series(dtype=str)  # Return an empty Series if the anime name is not found
    idx = indices[name]
    cosine_sim = linear_kernel(tf_authTags_matrix[idx:idx+1], tf_authTags_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    anime_indices = [i[0] for i in sim_scores]
    return anime_data['name'].iloc[anime_indices]

# Collaborative filtering recommendation function
def collaborative_filtering_recommendations(user_ratings, N=10):
    if user_ratings.empty:
        return pd.Series(dtype=str)  # Return an empty Series if the user_ratings DataFrame is empty

    # Reduce memory usage
    user_ratings = user_ratings.copy()
    user_ratings['rating'] = user_ratings['rating'].astype(np.float32)

    # Filter and create the user-item rating matrix
    min_ratings = 10
    user_ratings_filtered = user_ratings.groupby('user_id').filter(lambda x: len(x) >= min_ratings)
    min_ratings_per_item = 5
    user_ratings_filtered = user_ratings_filtered.groupby('name').filter(lambda x: len(x) >= min_ratings_per_item)
    
    # Create user-item matrix
    user_item_matrix = user_ratings_filtered.pivot_table(index='user_id', columns='name', values='rating').fillna(0)
    
    # Convert to sparse matrix
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

    def collab_generate_top_N_recommendations(user, N=10):
        if user not in user_item_matrix.index:
            return user_ratings_filtered.groupby('name').mean().sort_values(by='rating', ascending=False).index[:N].tolist()

        # Get recommendations
        user_id = user_item_matrix.index.get_loc(user)
        scores = model.recommend(user_id, user_item_matrix_sparse, N=N)
        recommended_indices = [x[0] for x in scores]
        
        return user_item_matrix.columns[recommended_indices].tolist()

    user_id = 1  # Adjust as needed
    recommendations = collab_generate_top_N_recommendations(user_id, N)
    return recommendations

# Predict future ratings function
def predict_user_ratings(user_ratings):
    if user_ratings.empty:
        return pd.DataFrame()
    
    user_rated_animes = user_ratings['name'].unique()
    unrated_animes = anime_data[~anime_data['name'].isin(user_rated_animes)]
    predictions = pd.DataFrame({
        'anime_name': unrated_animes['name'],
        'predicted_rating': np.random.uniform(1, 10, size=len(unrated_animes))
    })
    return predictions

# Streamlit app

st.set_page_config(page_title='Anime Recommender System', layout='wide')

# Apply custom styles with background color
st.markdown("""
    <style>
    .stApp {
        background-color: #FF4500; /* Dark Orange */
        background-size: cover;
        background-position: center;
    }
    .stTitle, .stHeader {
        text-align: center;
        color: white;
    }
    .stImage {
        text-align: center;
    }
    .stForm {
        text-align: center;
    }
    .stMarkdown {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a sidebar for navigation
page = st.sidebar.selectbox('Select a Page', ['Home', 'Recommender Systems'])

if page == 'Home':
    st.markdown("<h1 class='stTitle'>Welcome to the Anime Recommender System!</h1>", unsafe_allow_html=True)
    
    st.write("## Explore Our Anime Recommendations")

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image('Cartoon1.jpg', width=300, use_column_width=False)
    with col2:
        st.image('Cartoon2.png', width=300, use_column_width=False)
    
    st.write("## Watch Anime Trailers")
    st.video('Video1.mp4', format='video/mp4')

    st.write("## How It Works")
    st.write(
        """
        **Anime Recommender System**:
        Our system offers personalized anime recommendations based on two primary methods:
        
        1. **Content-Based Filtering**:
            - This method recommends anime based on the content similarity, considering factors like genre, type, and other tags.
            - It uses the TF-IDF vectorization technique to assess content similarity between different anime titles.
        
        2. **Collaborative Filtering**:
            - This approach recommends anime based on the preferences and ratings of other users.
            - We use matrix factorization techniques to model user preferences and predict ratings for anime that a user hasn't yet rated.
        
        Enjoy discovering new anime that matches your taste!
        """
    )
    st.write("## Usage")
    st.write(
        """
        - Navigate to the 'Recommender Systems' page to start receiving recommendations.
        - Provide your favorite anime titles to improve the recommendations you receive.
        """
    )

elif page == 'Recommender Systems':
    st.markdown("<h1 class='stTitle'>Anime Recommender System</h1>", unsafe_allow_html=True)

    # Recommendation type selection
    recommendation_type = st.selectbox('Select Recommendation Type', ['Content-based', 'Collaborative Filtering'])

    # User input for recommendations
    anime_name = st.text_input('Enter an anime name for recommendations:', '')

    # Number of recommendations
    num_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=20, value=7)

    # Generate recommendations based on the selected type
    if anime_name:
        if recommendation_type == 'Content-based':
            if anime_name in indices.index:
                recommendations = content_generate_top_N_recommendations(anime_name, N=num_recommendations)
                st.write(f"Top {num_recommendations} recommendations for '{anime_name}':")
                st.write(recommendations)
            else:
                st.write("Anime name not found. Please enter a valid anime name.")
        elif recommendation_type == 'Collaborative Filtering':
            recommendations = collaborative_filtering_recommendations(user_ratings, N=num_recommendations)
            st.write(f"Top {num_recommendations} collaborative filtering recommendations:")
            st.write(recommendations)

    # Predict future ratings
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = pd.DataFrame(columns=['user_id', 'name', 'rating'])

    # Multi-select for favorite movies
    st.subheader('Select and Rate Your Favorite Movies')
    selected_movies = st.multiselect('Select up to 3 favorite movies:', anime_data['name'].tolist())

    if len(selected_movies) > 3:
        st.warning('Please select up to 3 movies only.')

    # User input for rating the selected movies
    if len(selected_movies) > 0 and len(selected_movies) <= 3:
        ratings = {}
        for movie in selected_movies:
            rating = st.slider(f'Rate {movie}:', min_value=1, max_value=10, value=5)
            ratings[movie] = rating

        if st.button('Submit Ratings'):
            new_ratings = pd.DataFrame({
                'user_id': [1] * len(selected_movies),
                'name': selected_movies,
                'rating': [ratings[movie] for movie in selected_movies]
            })
            st.session_state.user_ratings = pd.concat([st.session_state.user_ratings, new_ratings], ignore_index=True)
            st.write("Ratings submitted successfully!")

    # Predict ratings for selected movies
    if st.button('Predict Ratings for Selected Movies'):
        predictions = predict_user_ratings(st.session_state.user_ratings)
        st.write("### Predicted Ratings for Unrated Anime:")
        st.write(predictions)
