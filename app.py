import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and vectorizer 
model = joblib.load('P:/ML_PROJECTS/movies/notebook/model.pkl') 
vectorizer = joblib.load('P:/ML_PROJECTS/movies/notebook/vectorizer.pkl') 
scaler = joblib.load('P:/ML_PROJECTS/movies/notebook/scaler.pkl')

st.title('Movie Audience Rating Prediction')

# Input fields for movie features
genre = st.selectbox("Select Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Unknown"])
runtime = st.number_input('Runtime (minutes)', min_value=0, value=240)
tomatometer_rating = st.number_input('Tomatometer Rating', min_value=0, max_value=100, value=90)
tomatometer_count = st.number_input('Tomatometer Count', min_value=0, value=200)


if st.button('Predict'):
    # Create a DataFrame from the input features
    new_movie_data = pd.DataFrame({
        'genre': [genre],
        'runtime_in_minutes': [runtime],
        'tomatometer_rating': [tomatometer_rating],
        'tomatometer_count': [tomatometer_count]
    })

    # Preprocess the new movie data
    X_text_new = new_movie_data[['genre']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    X_tfidf_new = vectorizer.transform(X_text_new)
    X_numerical_new = new_movie_data[['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count']]
    X_combined_new = np.hstack((X_tfidf_new.toarray(), X_numerical_new))
    X_combined_new = scaler.transform(X_combined_new)

    # Make prediction
    prediction = model.predict(X_combined_new).round(1)[0]
    
    # Display the result
    st.metric(label="Predicted Audience Rating", value=f"{prediction}")

