import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from scipy.sparse.linalg import svds
from skimage import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import AutoMinorLocator
from sparse_dot_mkl import dot_product_mkl

class BookLoader():

    def __init__(self, user_books):
        print("Loading book data...")
        wd = os.getcwd()
        self.books = pd.read_csv(wd + "/data/goodreads_books.csv")
        self.genres = pd.read_csv(wd + "/data/inferred_genres.csv")
        self.genre_descriptors = pd.read_csv(wd + "/data/inferred_genre_top_words.csv")
        self.target_books = user_books
        self.reviews = sparse.load_npz(wd + "/data/user_reviews.npz")
        self.user_index = pd.read_csv(wd + "/data/user_index_for_sparse_matrix.csv").rename(columns={"0":"user_id"})
        self.book_index = pd.read_csv(wd + "/data/book_index_for_sparse_matrix.csv").rename(columns={"0":"book_id"})        

class BookRecommender():

    def __init__(self, user_books):
        
        data = BookLoader(user_books)
        self.all_books = data.books
        self.target_books = data.target_books
        self.genres = data.genres
        self.genre_descriptors = data.genre_descriptors
        self.reviews = data.reviews 
        self.user_index = data.user_index 
        self.book_index = data.book_index        
        self.target = data.reviews.shape[0]

        self.prep_data()
        self.find_neighbors()
        self.get_recs()
        self.similar_readers_most_popular()
        self.similar_readers_top_rated()

    def prep_data(self):
        """
        Description:
            Abc

        Returns:
            _type_: _description_
        """
        print("Prepping data...")

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
        self.reviews = sparse.vstack([self.reviews, my_books])
        self.reviews = sparse.csc_matrix(self.reviews)

        # Normalize reviews within readers
        norm = Normalizer()
        self.row_norms = np.sqrt(np.sum(self.reviews.power(2), axis=1)) # save row norms to un-normalzie later
        self.reviews = norm.fit_transform(self.reviews) 
        self.all_books = df_books

    
    def find_neighbors(self, n_neighbors=3000):
        """
        Description:
            Abc

        Returns:
            _type_: _description_
        """

        print("Finding similar readers...")

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
        reader = self.reviews[self.target,:].toarray()
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

        
        # Get target user's top genres
        self.genre_ranking = pd.DataFrame(target_user_ratings.loc[:, "Genre_1":].sum(axis=0).sort_values(ascending=False))
        self.target_user_ratings = target_user_ratings
        self.neighbor_user_ratings = neighbor_user_ratings


    def get_recs(self, min_rating=3.5):
        """
        Abc
        """

        print("Generating recommendations...")

        # Get unique users and books to slice reviews
        neighbor_index = self.neighbor_user_ratings["uid"].unique()
        neighbor_index = np.append(neighbor_index, self.target)
        neighbor_book_index = self.neighbor_user_ratings["book_index"].unique()
        neighbor_book_index = np.append(neighbor_book_index, self.target_user_ratings["book_index"].unique())

        # Slice reviews to make User Ratings Matrix
        R = self.reviews[:, neighbor_book_index]
        R = R[neighbor_index, :]
        
        # Decompose user ratings matrix R with SVD
        U, sigma, Vt = svds(R, k=42)
        sigma = np.diag(sigma)

        # Convert to sparse matrix
        U = sparse.csr_matrix(U)
        sigma = sparse.csr_matrix(sigma)
        Vt = sparse.csc_matrix(Vt)

        # Get prediction
        all_user_predicted_ratings = dot_product_mkl(dot_product_mkl(U, sigma, dense=False), Vt, dense=True)
        df_preds = pd.DataFrame(
                        all_user_predicted_ratings, columns=neighbor_book_index, index=neighbor_index
                        ).reset_index()
        target_pred_books = df_preds[df_preds["index"] == self.target].columns[1:]
        target_pred_ratings = df_preds[df_preds["index"] == self.target].values[0][1:] * float(self.row_norms[self.target])

        # Put into df with relevant info from df_books
        top_preds = pd.DataFrame({"book_index":target_pred_books, "predicted_rating":target_pred_ratings})\
                        .sort_values(by="predicted_rating", ascending=False)\
                        .merge(self.book_index.reset_index(), left_on="book_index", right_on="index")\
                        .merge(
                            self.all_books[["book_id", "title", "avg_rating", "ratings_count", "year", "main_genre","url","author"]],
                            on="book_id"
                        )\
                        .drop(columns=["index"])

        # Filter out already read books
        top_preds = top_preds[~top_preds["book_index"].isin(self.target_user_ratings["book_index"].unique())]
        top_preds.drop(["book_index"], axis=1, inplace=True)

        # Add predicted rating column
        top_preds["predicted_rating"] = round(top_preds["predicted_rating"] + self.neighbor_user_ratings["user_rating"].mean(), 2)

        # Get genre descriptions
        self.genre_descriptors["genre_num"] = self.genre_descriptors["genre_string"].apply(lambda x: int(x.split(":")[0].split(" ")[1]))

        top_preds = pd.merge(top_preds, self.genre_descriptors, left_on="main_genre", right_on="genre_num")
        top_preds.rename(columns={"genre_string":"genre_name"}, inplace=True)
        
        top_preds = top_preds[["book_id", "title","avg_rating","predicted_rating","ratings_count","year","url", "genre_name","author"]]\
                .query("avg_rating > @min_rating").sort_values(by="predicted_rating", ascending=False)

        self.recs = top_preds
        print("Recommendations ready!")

    def similar_readers_most_popular(self):
        """
        ABC
        """
        others = self.neighbor_user_ratings
        others = pd.merge(others.groupby("book_id")["user_rating"].mean()\
                                .reset_index().rename(columns={"user_rating":"similar_usr_avg"}),
                        others,
                        on="book_id")
        others["similar_usr_avg"] = others["similar_usr_avg"].round(2)

        popular_recs = others.query("ratings_count > 100")\
            .groupby(["title", "avg_rating", "similar_usr_avg", "ratings_count", "year", "url", "author", "main_genre"])["book_id"]\
            .count().reset_index()\
            .rename(columns={"book_id":"%_similar_usr_read"})

        popular_recs["%_similar_usr_read"] = (popular_recs["%_similar_usr_read"] / 
                                                others["uid"].nunique()).map('{:.1%}'.format)
        
        # Get genre descriptions
        self.genre_descriptors["genre_num"] = self.genre_descriptors["genre_string"].apply(lambda x: int(x.split(":")[0].split(" ")[1]))

        popular_recs = pd.merge(popular_recs, self.genre_descriptors, left_on="main_genre", right_on="genre_num")
        popular_recs.rename(columns={"genre_string":"genre_name"}, inplace=True)
        
        self.similar_readers_popular = popular_recs[["title","avg_rating","similar_usr_avg", "ratings_count","year","%_similar_usr_read","url","author","genre_name"]].sort_values(by=["%_similar_usr_read","avg_rating"], ascending=False)
    

    # Function to show top rated among similar readers
    def similar_readers_top_rated(self, n=10):
        """
        ABC
        """
        others = self.neighbor_user_ratings
        others = pd.merge(others.groupby("book_id")["user_rating"].mean()\
                                .reset_index().rename(columns={"user_rating":"similar_usr_avg"}),
                        others,
                        on="book_id")
        others["similar_usr_avg"] = others["similar_usr_avg"].round(2)
        
        min_neighbor_ratings = np.max([others["uid"].nunique() / 300, 5])

        highest_rated_recs = others.query(
                    "ratings_count > 100 & YA == 0")\
            .groupby(["title", "avg_rating", "similar_usr_avg", "ratings_count", "year", "url", "author", "main_genre"])["uid"]\
            .count().reset_index()\
            .query("uid >= @min_neighbor_ratings")\
            .drop(columns="uid")
        
        # Get genre descriptions
        self.genre_descriptors["genre_num"] = self.genre_descriptors["genre_string"].apply(lambda x: int(x.split(":")[0].split(" ")[1]))

        highest_rated_recs = pd.merge(highest_rated_recs, self.genre_descriptors, left_on="main_genre", right_on="genre_num")
        highest_rated_recs.rename(columns={"genre_string":"genre_name"}, inplace=True)

        self.similar_readers_highly_rated = highest_rated_recs.sort_values(by=["similar_usr_avg", "avg_rating"], ascending=False)
    

    def find_similar_books_to(self, title, min_rating=3.5, n=25):
        """
        ABC
        """

        title_search = title.lower()

        # Instantiate KNN
        nn_model = NearestNeighbors(
            metric="cosine",
            algorithm="auto",
            n_neighbors=n,
            n_jobs=-1
        )

        # Fit to sparse matrix
        nn_model.fit(self.reviews.T)

        # Feed in book and get neighbors and distances
        if self.all_books[self.all_books["title"].str.lower().str.contains(title_search)].shape[0] > 1:
            print("Warning: more than 1 book matches title search. Returning results for top match")
            
        target_book_id = self.all_books[self.all_books["title"].str.lower().str.contains(title_search)]\
                            .sort_values(by=["ratings_count", "avg_rating"], ascending=False).iloc[0,:]["book_id"]
        target_book_index = self.book_index[self.book_index["book_id"]==target_book_id].index[0]
        target_book = self.reviews.T[target_book_index,:].toarray()

        dists, neighbors = nn_model.kneighbors(target_book, return_distance=True)

        similar_books = pd.DataFrame(
            [pd.Series(neighbors.reshape(-1)), pd.Series(dists.reshape(-1))]).T.rename(
                columns={0:"book", 1:"distance"}
        )

        similar_books = pd.merge(similar_books, self.book_index, left_on="book", right_index=True)
        similar_books = similar_books[~similar_books["book_id"].isin(self.target_user_ratings["book_id"])]
        similar_books = pd.merge(similar_books["book_id"], self.all_books, on="book_id")

        # Filter out later volumes in series using regex pattern
        regex1 = r"#(?:[2-9]|[1-9]\d+)"
        regex2 = r"Vol. (?:[0-9]|[1-9]\d+)"
        regex3 = r"Volume (?:[0-9]|[1-9]\d+)"
        similar_books = similar_books[~similar_books["title"].str.contains(regex1)]
        similar_books = similar_books[~similar_books["title"].str.contains(regex2)]
        similar_books = similar_books[~similar_books["title"].str.contains(regex3)]
        similar_books = similar_books[~similar_books["title"].str.contains("#1-")]
        
        similar_books = similar_books.iloc[1:,:].query('avg_rating >= @min_rating & ratings_count > 500`')[
            ["title","author","avg_rating", "ratings_count", "year", "url"]
        ]

        return similar_books
    

    def plot_top_books(self):
        """
        Plots book cover images for top recommendations in top 5 genres

        Args:
            preds: prediction df yielded by get_recs()
            books: dataframe with all book info
            target_ratings: target user's book ratings

        Returns:
            None
        """
        # Get image_url for top preds
        plot_pred = pd.merge(
                        self.recs[["book_id", "predicted_rating"]], 
                        self.all_books[["book_id", "image_url", "main_genre"]], 
                        on="book_id"
                    )

        # For each of the top 10 genres, get top 5 books
        top_genres = pd.DataFrame(self.target_user_ratings.loc[:, "Genre_1":].sum(axis=0)\
                                                .sort_values(ascending=False)).rename(columns={0:"target"})[0:5]

        # Function to turn image url into image 
        def getImage(path, zoom=0.3):
            return OffsetImage(io.imread(path), zoom=zoom)

        # Plot
        fig, ax = plt.subplots(figsize=(13,7))
        for i in range(len(top_genres)): # For each of top genres
            genre = top_genres.index[i]
            g = float(genre[6:]) # Get genre number
            
            # Slice plot pred to genre and valid image url
            books_to_plot = plot_pred[ 
                (plot_pred["main_genre"] == g) & (plot_pred["image_url"].str.contains("images.gr-"))
                ].head(5)
            
            paths = [url for url in books_to_plot["image_url"]] # Get image urls
            x = [i * 10 + 10 for x in range(5)] # Set genre bucket
            y = [y for y in books_to_plot["predicted_rating"]] # get predicted rating as y

            # Plot
            ax.scatter(x,y,alpha=0) 

            # Plot image at xy
            for x0, y0, path in zip(x, y, paths):
                ab = AnnotationBbox(getImage(path), (x0 + np.random.uniform(-3.5,3.5), y0), frameon=True, pad=0.3)
                ax.add_artist(ab)
            
        plt.ylabel("Recommendation Strength", fontsize=12)
        plt.xlabel("\nGenre Grouping", fontsize=12)
        plt.title("Top Recommendations by Genre", y=1.02, fontsize=14)
        plt.xlim((5, 55))
        ax.spines[['right', 'top', 'left']].set_visible(False)
        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        ax.yaxis.set_ticklabels([])
        ax.yaxis.grid(True, alpha=0.3) # Create y gridlines
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.grid(which="minor", visible=True, alpha=0.8) # Create x gridlines
        plt.show()


    # Function to plot neighbors' and target's top genres
    def plot_top_genres(self):
        
        others = self.neighbor_user_ratings
        target = self.target_user_ratings

        # Get genre rankings for target and neighbors
        target_genre_ranking = pd.DataFrame(target.loc[:, "Genre_1":].sum(axis=0)\
                                                .sort_values(ascending=False)).rename(columns={0:"target"})
        target_genre_ranking = target_genre_ranking.div(target_genre_ranking.sum(axis=0), axis=1)

        neighbor_genre_ranking = pd.DataFrame(others.loc[:, "Genre_1":].sum(axis=0)\
                                                .sort_values(ascending=False)).rename(columns={0:"neighbor"})
        neighbor_genre_ranking = neighbor_genre_ranking.div(neighbor_genre_ranking.sum(axis=0), axis=1)

        genre_rankings = pd.merge(
                target_genre_ranking, neighbor_genre_ranking, left_index=True, right_index=True
                ).reset_index()

        # Plot target genre pref
        fig, ax = plt.subplots(figsize=(10,10))
        sns.scatterplot(
            data=genre_rankings,
            y='index',
            x='target',
            s=150,
            edgecolors='black',
            color="mediumslateblue",
            linewidths = 0.75,
            label='You',
            zorder=2,
            )

        # Plot neighbors genre pref
        sns.scatterplot(
            data=genre_rankings,
            y='index',
            x='neighbor',
            label='Similar Readers',
            color="darkgray",
            s=150,
            zorder=3
            )

        # Iterate through each genre and plot line connecting 2 points
        for ind in list(genre_rankings['index']):
        
            # Plot line connecting points
            plt.plot([genre_rankings[genre_rankings['index']==ind]['target'],
                        genre_rankings[genre_rankings['index']==ind]['neighbor']],
                        [ind, ind],
                        color='#565A5C',
                        alpha=0.2,                    
                        # linestyle=(0, (1,1)),
                        linewidth=6.5,
                        zorder=1
                        )

        # Set chart details
        plt.legend(bbox_to_anchor=(1,1), loc="upper left", borderpad=1)
        ax.yaxis.grid(True, alpha=0.4) # Create y gridlines
        ax.xaxis.grid(True, alpha=0.4) # Create x gridlines
        plt.xlabel("Genre preference")
        plt.ylabel(None)
        plt.title('Your Top Genres Compared with Similar Readers', fontsize=14)
        plt.show()
