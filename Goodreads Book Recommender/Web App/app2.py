from flask import Flask, render_template, request, jsonify
import pandas as pd  
from recommender_classes import BookRecommender, BookLoader

app = Flask(__name__)

recs_obj = []


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    csv_file = request.files['csvFile']
    make_recs(csv_file)          
    recs_dict, popular_dict, top_rated_dict, my_ratings_dict = process_recs(recs_obj[0])     
    
    return jsonify([recs_dict, popular_dict, top_rated_dict, my_ratings_dict]) 
       

def make_recs(csv):    
    try:        
        df = pd.read_csv(csv)        
        recommendations = BookRecommender(df)
        recs_obj.append(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)})

def process_recs(rec):
    try:
        recs = rec.recs.head(3500)
        popular = rec.similar_readers_popular.head(1000)
        top_rated = rec.similar_readers_highly_rated.head(3000)
        my_ratings = rec.target_user_ratings

        for d in [recs, popular, top_rated, my_ratings]:
            d['year'] = d['year'].fillna(0)
            
        recs = recs[["title", "author", "avg_rating", "predicted_rating", "url", "genre_name"]]
        recs.rename(columns={"avg_rating":"average","predicted_rating":"predicted","genre_name":"genre"}, inplace=True)
        
        popular = popular[["title", "author", "avg_rating", "similar_usr_avg", "%_similar_usr_read", "url", "genre_name"]]
        popular.rename(columns={"avg_rating":"average","similar_usr_avg":"similarReadersAvg","%_similar_usr_read":"similarRead%","genre_name":"genre"}, inplace=True)
        
        top_rated = top_rated[["title", "author", "avg_rating", "similar_usr_avg", "url", "genre_name"]]
        top_rated.rename(columns={"avg_rating":"average","similar_usr_avg":"similarReadersAvg","genre_name":"genre"}, inplace=True)

        my_ratings = my_ratings[["title", "author", "avg_rating", "user_rating", "url"]]
        my_ratings["user_rating"] = my_ratings["user_rating"] * (5 / my_ratings["user_rating"].max())
        my_ratings.sort_values(by=["user_rating","avg_rating"], ascending=False, inplace=True)
        my_ratings.rename(columns={"avg_rating":"average", "user_rating":"myRating"}, inplace=True)
        
        # Create a dictionary that includes the order of columns
        recs_dict = {
            'columns': list(recs.columns),
            'data': recs.to_dict(orient='records')
        }

        popular_dict = {
            'columns': list(popular.columns),
            'data': popular.to_dict(orient='records')
        }

        top_rated_dict = {
            'columns': list(top_rated.columns),
            'data': top_rated.to_dict(orient='records')
        }

        my_ratings_dict = {
            'columns': list(my_ratings.columns),
            'data': my_ratings.to_dict(orient='records')
        }        
        
        # Return the result to the front-end
        return (recs_dict, popular_dict, top_rated_dict, my_ratings_dict)

    except Exception as e:
        return jsonify({'error': str(e)})
    
    
@app.route('/similar_books', methods=["POST"])
def similar_books():
    try:        
        recs = recs_obj[0]        
        title = request.get_json()                
        
        similar_books = recs.find_similar_books_to(title) 
        
        similar_books.drop(columns=["ratings_count", "year"], inplace=True)        
        similar_books.rename(columns={"avg_rating":"average"}, inplace=True)
        
        similar_books_dict = {
            'columns': list(similar_books.columns),
            'data': similar_books.to_dict(orient='records')
        }

        return jsonify(similar_books_dict)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)