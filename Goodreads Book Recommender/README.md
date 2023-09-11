### Goodreads Book Recommender

**Description:**
Uses [goodreads data](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0) scraped by Mengting Wan and Julian McAuley at UCSD to build a recommender system using three methods:
1. Collaborative filtering with KNN to suggest popular and highly rated books among a similar set of readers to the target reader
2. Matrix Factorization with SVD of a user-rating matrix to predict ratings for unread books
3. Matrix Factorization with gradient descent by alternating least squares (ALS) to predict ratings for unread books

Scripts include:
* **00_prep_goodreads_data.ipynb** - imports, cleans, and prepares the UCSD data for later steps
* **01_infer_genres.ipynb** - performs topic modeling via Latent Dirichlet Allocation (LDA) to infer genres based on each book's description text. These genres are used for making recommendations in the next step
* **02_book_recommender.ipynb** - generates book recommendations with user-user similarity via KNN and user-item rating predictions via matrix factorization

![output2](https://github.com/mraottth/projects/assets/64610726/02633d23-3938-4252-a409-92b5c7b519a5)


**Filetree:**
```
├── data
│   ├── book_index_for_sparse_matrix.csv
|   ├── goodreads_books.csv
|   ├── goodreads_library_export.csv
|   ├── inferred_genres.csv
|   ├── user_index_for_sparse_matrix.csv
│   └── user_reviews.npz
│── book_recommender.ipynb
│── infer_genres.ipynb
└── prep_goodreads_data.ipynb
```

___
