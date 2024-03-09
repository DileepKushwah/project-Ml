#!/usr/bin/env python3

"""
This file is part of the Hollywood Movie Success Predictor project.

The Hollywood Movie Success Predictor project is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Hollywood Movie Success Predictor project is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the Hollywood Movie Success Predictor project.  If not, see <https://www.gnu.org/licenses/>.
"""



import pickle
import pandas as pd
import streamlit as st
import requests

# Load movie data and similarity matrix
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

OMDB_API_KEY = "9638bcac"

def fetch_poster(movie_title):
    try:
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if data['Response'] == 'True' and 'Poster' in data:
            return data['Poster']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print("Error fetching poster:", e)
        return None

def recommend(movie):
    recommended_movies = []  # Initialize list to store recommended movies
    recommended_movies_poster = []  # Initialize list to store recommended movies posters

    # Retrieve index of selected movie
    movie_index = movies[movies['title'] == movie].index[0]
    # Retrieve similarity scores for the selected movie
    distances = similarity[movie_index]
    # Sort similarity scores and get top 5 similar movies (excluding the selected movie itself)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    # Retrieve titles of recommended movies and their corresponding poster URLs
    for i in movie_list:
        recommended_movie = movies.iloc[i[0]].title
        recommended_movies.append(recommended_movie)
        poster_url = fetch_poster(recommended_movie)
        recommended_movies_poster.append(poster_url)
    return recommended_movies, recommended_movies_poster

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #E50914;'>Movie Recommender System</h1>", unsafe_allow_html=True)

# Dropdown to select a movie
selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values,
    key='movie-dropdown',
    help="Select a movie from the dropdown."
)

# Button to trigger recommendation
if st.button('Show Recommendations', key='recommend-button'):
    recommendations, recommendations_poster = recommend(selected_movie_name)
    # Display recommended movies
    if recommendations:
        cols = st.columns(5)
        for idx, (movie, poster_url) in enumerate(zip(recommendations, recommendations_poster), start=0):
            cols[idx % 5].write(f"{idx+1}. {movie}")
            if poster_url:
                cols[idx % 5].image(poster_url, caption=movie, width=120)
    else:
        st.write("No recommendations found for this movie.")


#STEP 1= for run this program Create New terminal 
#STEP 2= write IN TERMINAL = streamlit run app.py

