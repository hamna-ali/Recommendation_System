import pandas as pd

data = {
    'User': ['A','A','A','B','B','C','C','C','D'],
    'Item': ['Movie1','Movie2','Movie3','Movie1','Movie3','Movie2','Movie3','Movie4','Movie4'],
    'Rating': [5,4,3,4,5,2,5,4,5]
}

df = pd.DataFrame(data)
print(df)


user_item_matrix = df.pivot_table(
    index='User',
    columns='Item',
    values='Rating'
)

print(user_item_matrix)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

user_item_filled = user_item_matrix.fillna(0)

similarity_matrix = cosine_similarity(user_item_filled)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print(similarity_df)


def recommend_items(target_user, n=2):
    similar_users = similarity_df[target_user].sort_values(ascending=False)
    similar_users = similar_users.drop(target_user)

    weighted_ratings = pd.Series(dtype=float)

    for user, similarity in similar_users.items():
        user_ratings = user_item_matrix.loc[user]
        weighted_ratings = weighted_ratings.add(
            user_ratings * similarity,
            fill_value=0
        )

    already_rated = user_item_matrix.loc[target_user].dropna().index
    recommendations = weighted_ratings.drop(already_rated)

    return recommendations.sort_values(ascending=False).head(n)


print("Recommendations for User A:")
print(recommend_items('A'))
