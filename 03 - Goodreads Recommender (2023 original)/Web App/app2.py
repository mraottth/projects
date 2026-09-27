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
    
    if csv_file.filename.endswith('.csv'):        
        make_recs(csv_file)      
        process_recs(recs_obj[0])  
    else:
        return jsonify({'error': 'Invalid file format'})    

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
        for d in [recs, popular, top_rated]:
            d['year'] = d['year'].fillna(0)
            
        recs = recs[["title", "author", "avg_rating", "predicted_rating", "url", "genre_name"]]
        recs.rename(columns={"avg_rating":"average","predicted_rating":"predicted","genre_name":"genre"}, inplace=True)
        
        popular = popular[["title", "author", "avg_rating", "similar_usr_avg", "%_similar_usr_read", "url", "genre_name"]]
        popular.rename(columns={"avg_rating":"average","similar_usr_avg":"similarReadersAvg","%_similar_usr_read":"similarRead%","genre_name":"genre"}, inplace=True)
        
        top_rated = top_rated[["title", "author", "avg_rating", "similar_usr_avg", "url", "genre_name"]]
        top_rated.rename(columns={"avg_rating":"average","similar_usr_avg":"similarReadersAvg","genre_name":"genre"}, inplace=True)

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
        
        # Return the result to the front-end
        return jsonify([recs_dict, popular_dict, top_rated_dict])

    except Exception as e:
        return jsonify({'error': str(e)})
    
    
@app.route('/similar_books', methods=["POST"])
def similar_books(rec):
    try:
        title = request.get_json()        
        similar_books = rec.find_similar_books_to(title) 
        return jsonify(similar_books)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)