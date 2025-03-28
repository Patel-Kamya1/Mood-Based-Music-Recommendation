import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
try:
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Load dataset
try:
    songs_df = pd.read_csv("clustered_df.csv")
except Exception as e:
    st.error(f"Error loading clustered_df.csv: {e}")
    st.stop()

# Page title
st.title("🎧 Mood-Based Music Recommendation System")
st.subheader("Adjust your current mood below:")

# Sliders for 7 features used during training
valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.3)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)

# Recommend songs
if st.button("🎶 Recommend Songs"):
    try:
        input_data = np.array([[valence, danceability, tempo, acousticness, liveness, speechiness, instrumentalness]])
        scaled_input = scaler.transform(input_data)

        cluster = kmeans.predict(scaled_input)[0]

        # Debug info (optional)
        st.write("🔍 Predicted Cluster:", cluster)

        if 'Cluster' not in songs_df.columns:
            st.error("❌ 'Cluster' column not found in dataset!")
            st.stop()

        recommended_songs = songs_df[songs_df['Cluster'] == cluster]

        st.success(f"✨ Songs from Mood Cluster: {cluster}")

        if recommended_songs.empty:
            st.warning("No songs found for this cluster.")
        else:
            # Show up to 10 songs
            for _, row in recommended_songs.sample(n=min(10, len(recommended_songs))).iterrows():
                st.write(f"🎵 **{row['name']}** by *{row['artists']}*")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
