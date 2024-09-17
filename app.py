import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data and feature vectors
df_filtered = pd.read_csv('recipes_data.csv')
vectors = np.load('feature_vectors.npy')

# Load the cosine similarity model (if needed)
# cosine_sim = joblib.load('cosine_similarity.pkl')

# Recommendation function
def get_recommendations(recipe_name: str, vector: np.array, attributes: str, num_returned_recipes: int):
    try:
        # Get the index of the recipe that matches the `recipe_name`
        idx = df_filtered[df_filtered['name'] == recipe_name].index[0]

        # Get the feature vector for the target recipe
        target_vector = vector[idx]

        # Calculate cosine similarity between the target recipe and all other recipes
        cosine_sim = cosine_similarity(target_vector.reshape(1, -1), vector).flatten()

        # Get the similarity scores
        sim_scores = list(enumerate(cosine_sim))

        # Sort the recipes based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the `num_returned_recipes` most similar recipes
        sim_scores = sim_scores[1:num_returned_recipes + 1]

        # Get the recipe indices
        recipe_indices = [i[0] for i in sim_scores]

        # Find the top `num_returned_recipes` most similar recipes
        recommendations = df_filtered.iloc[recipe_indices]
        
        return recommendations
    except IndexError:
        return None

# Streamlit App
st.title("Recipe Recommendation System")

# Select the number of recommendations
num_recommendations = st.slider("Select the number of recommendations:", 1, 10, 5)

# Select recipe attributes to base recommendations on (this could be used to tweak recommendation logic)
attributes = st.selectbox("Select recipe attribute to focus on:", ['Ingredients', 'Steps', 'Calories', 'Minutes'])

# Input for recipe name
recipe_name = st.text_input("Enter the recipe name for which you want recommendations:")

if st.button("Get Recommendations"):
    if recipe_name:
        recommendations = get_recommendations(recipe_name, vectors, attributes, num_recommendations)

        if recommendations is not None:
            st.write(f"Recommendations based on the recipe: **{recipe_name}**")
            for i, row in recommendations.iterrows():
                st.subheader(f"Recommendation #{i+1}")
                st.write(f"**Recipe name**: {row['name']}")
                st.write(f"**Ingredients**: {row['ingredients']}")
                st.write(f"**Steps**: {row['steps']}")
                st.write(f"**Minutes**: {row['minutes']}")
                st.write(f"**Calories**: {row['calories']}")
                st.write("---")
        else:
            st.write("Recipe not found in the database. Please try another recipe name.")
    else:
        st.write("Please enter a recipe name to get recommendations.")

# Run the app using `streamlit run app.py`
