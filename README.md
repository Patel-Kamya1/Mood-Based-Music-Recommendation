# Mood-Based Music Recommendation System  

This is a music recommendation system that suggests songs based on your mood. It uses KMeans clustering to group songs and recommends tracks based on features like valence, danceability, tempo, and more.  

## Features  
- Suggests songs based on mood input  
- Uses KMeans clustering for recommendations  
- Adjustable sliders for setting mood parameters  
- Streamlit UI for easy interaction  

## Files  
- app.py → Streamlit app code  
- kmeans_model.pkl → Trained model  
- scaler.pkl → Scaler for preprocessing  
- clustered_df.csv → Dataset with song clusters
- data.csv → Original dataset used for clustering  

## How It Works  
- Enter your mood using sliders  
- The system scales the input and predicts a cluster  
- It recommends songs from the matching mood cluster  

