from recommender_classes import BookRecommender, BookLoader
import pandas as pd 
if __name__ == '__main__':
    wd = os.getcwd()
    df = pd.read_csv(wd + "/Web App/data/goodreads_library_export.csv")
    recs = BookRecommender(df)
