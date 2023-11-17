import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from IPython.display import display
from scipy.sparse.linalg import svds
from skimage import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import AutoMinorLocator


class Books():

    def __init__(self, all_books, target_books, genres, genre_descriptors, reviews, user_ind, book_ind):
        self.all_books = all_books
        self.target_books = target_books
        self.genres = genres
        self.genre_descriptors = genre_descriptors
        self.reviews = reviews 
        self.user_index = user_ind 
        self.book_index = book_ind        
        self.row_norms = None

    def prep_data(self):
        """
        Description:
            Abc

        Returns:
            _type_: _description_
        """
        # Join genres to books
        df_books = pd.merge(self.all_books, self.genres, how="left", on="book_id")

        # Filter out kids books
        df_books = df_books[df_books.loc[:,"children":"comic"].sum(axis=1) < 2]

        # Create reviews matrix that is less sparse than original
        in_df_books = self.book_index["book_id"].isin(df_books["book_id"])
        self.book_index = self.book_index[in_df_books].reset_index().drop("index", axis=1)
        self.reviews = self.reviews[:, in_df_books]

        # Filter to books target user has rated
        self.target_books = self.target_books[["Book Id", "My Rating", "Title"]]\
            .rename(columns={"Book Id":"book_id", "My Rating":"rating", "Title":"title"}).query("rating > 0")

        # First match on book_id
        my_books_1 = pd.merge(
                        self.target_books, df_books[["book_id", "title"]], 
                        how="inner", on="book_id", suffixes=["_mb","_dfb"]
                    ).rename(columns={"title_dfb":"title"}).drop("title_mb", axis=1)
        # Next match on title
        my_books_2 = pd.merge(
                        self.target_books[~self.target_books["book_id"].isin(my_books_1["book_id"])], # not matched on id
                        df_books[["book_id", "title", "ratings_count", "avg_rating"]], 
                        how="inner", on="title", suffixes=["_mb","_dfb"]
                    ).sort_values(["title", "ratings_count"], ascending=False)\
                    .drop_duplicates(subset=["title"])\
                    .rename(columns={"book_id_dfb":"book_id"})
        
        # Concat matching on book_id and matching on title
        my_books_3 = pd.concat([my_books_1, my_books_2[["book_id", "rating", "title"]]]).drop_duplicates(subset="title")

        # Reshape to sparse matrix dimensions
        my_books_4 = pd.merge(self.book_index, my_books_3, how="left", on="book_id").fillna(0.)
        my_books = np.array(my_books_4["rating"]).reshape(1,-1)

        # Add to sparse matrix
        self.reviews = sparse.vstack([self.reviews, self.target_books])
        self.reviews = sparse.csr_matrix(self.reviews)

        # Normalize reviews within readers
        norm = Normalizer()
        self.row_norms = np.sqrt(np.sum(self.reviews.power(2), axis=1)) # save row norms to un-normalzie later
        self.reviews = norm.fit_transform(self.reviews) 
        
    
    def find_neighbors(self, n_neighbors=3000):
        """
        Description:
            Abc

        Returns:
            _type_: _description_
        """

        # Get index for target user
        target = self.reviews.shape[0] - 1

        # Instantiate KNN
        nn_model = NearestNeighbors(
            metric="cosine",
            algorithm="auto",
            n_neighbors=n_neighbors,
            n_jobs=-1
        )

        # Fit to sparse matrix
        nn_model.fit(self.reviews)

        # Feed in user and get neighbors and distances
        reader = self.reviews[target,:].toarray()
        dists, neighbors = nn_model.kneighbors(reader, return_distance=True)

        similar_users = pd.DataFrame(
            [pd.Series(neighbors.reshape(-1)), pd.Series(dists.reshape(-1))]).T.rename(
                columns={0:"user", 1:"distance"}
        )

        # Get all books read by similar users
        book_ind = []
        book_rat = []
        uid = []
        target_user_books = []
        target_user_book_rat = []
        for nt in similar_users.itertuples():
            user = self.reviews[int(nt.user),:].toarray()
            book_inds = np.where(user[0] > 0)[0]
            ratings = user[0][np.where(user[0] > 0)[0]]
            for i in range(len(book_inds)):        
                book_ind.append(book_inds[i])
                book_rat.append(ratings[i] * float(self.row_norms[int(nt.user)])) 
                uid.append(nt.user)    
                if nt.distance < 0.000000001:
                    target_user_books.append(book_inds[i])
                    target_user_book_rat.append(ratings[i])

        neighbor_user_ratings = pd.DataFrame([uid, book_ind, book_rat]).T.rename(
                                    columns={0:"uid",1:"book_index",2:"user_rating"}
                                )
        
        # Join overall rating for each book
        neighbor_user_ratings = pd.merge(
                                    self.book_index.reset_index(), neighbor_user_ratings, 
                                    how="inner", left_on="index", right_on="book_index"
                                )
        neighbor_user_ratings = pd.merge(neighbor_user_ratings, self.all_books, how="inner", on="book_id")

        # Filter out books target reader has already read
        neighbor_user_ratings = neighbor_user_ratings[~neighbor_user_ratings["book_index"].isin(target_user_books)]
        neighbor_user_ratings.drop(["index"], axis=1, inplace=True)

        # Filter out later volumes in series using regex pattern
        regex1 = r"#(?:[2-9]|[1-9]\d+)"
        regex2 = r"Vol. (?:[0-9]|[1-9]\d+)"
        regex3 = r"Volume (?:[0-9]|[1-9]\d+)"
        neighbor_user_ratings = neighbor_user_ratings[~neighbor_user_ratings["title"].str.contains(regex1)]
        neighbor_user_ratings = neighbor_user_ratings[~neighbor_user_ratings["title"].str.contains(regex2)]
        neighbor_user_ratings = neighbor_user_ratings[~neighbor_user_ratings["title"].str.contains(regex3)]
        neighbor_user_ratings = neighbor_user_ratings[~neighbor_user_ratings["title"].str.contains("#1-")]

        # Get target user's ratings
        target_user_ratings = pd.DataFrame(
                                [target_user_books, target_user_book_rat]).T.rename(
                                    columns={0:"book_index",1:"user_rating"}
                            )
        target_user_ratings = pd.merge(
                                self.book_index.reset_index(), target_user_ratings,
                                how="inner", left_on="index", right_on="book_index"
                            )
        target_user_ratings = pd.merge(target_user_ratings, self.all_books, how="inner", on="book_id")    

        return neighbor_user_ratings, target_user_ratings, dists

    
    def top_recs_by_genre(self, )







# Function to show top recs for each genre
def top_recs_by_genre(target, others, preds, n_genres=5, min_score=3.5, min_ratings=100, n_recs=20):

    genre_ranking = pd.DataFrame(target.loc[:, "Genre_1":].sum(axis=0).sort_values(ascending=False))
    others_ratings = others.groupby("book_id")["user_rating"].mean()\
                                                .reset_index()\
                                                .rename(columns={"user_rating":"similar_usr_avg"})
    genres = []
    results = []
    # Loop through genres in descending relevance order and print top recs
    for genre in genre_ranking.index[0:n_genres]:        
        
        g = float(genre[6:])
        highest_rated_recs_genre = preds.query("main_genre == @g").sort_values(by="predicted_rating", 
                                                                                   ascending=False)                

        highest_rated_recs_genre = highest_rated_recs_genre.query(
                                        "ratings_count > @min_ratings & avg_rating > @min_score"
                                    )
        
        highest_rated_recs_genre = pd.merge(highest_rated_recs_genre, others_ratings, how="left", on="book_id")
        
        highest_rated_recs_genre["similar_usr_avg"] = highest_rated_recs_genre["similar_usr_avg"].round(2)
        
        cols = ["title", "avg_rating", "similar_usr_avg", "ratings_count", "year", "url"]
        highest_rated_recs_genre = highest_rated_recs_genre[cols]

        # Print genre name and descriptor
        for nt in genre_descriptors.itertuples():
            genre_rep = genre.replace("_"," ")
            if nt.genre_string[0:len(f"{genre}:")] == f"{genre_rep}:":
                gs = (nt.genre_string)
        
        genres.append(gs)
        results.append(highest_rated_recs_genre.head(50))

        display(highest_rated_recs_genre.head(50))
    