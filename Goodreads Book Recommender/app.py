from flask import Flask, render_template, request, jsonify
import pandas as pd  
from recommender_classes import BookRecommender, BookLoader

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_route():
    try:
        csv_file = request.files['csvFile']

        # Load CSV data into a pandas DataFrame
        df = pd.read_csv(csv_file)

        recommendations = BookRecommender(df)
        recs = recommendations.recs
        recs['year'] = recs['year'].fillna(0)
        recs = recs[["title", "avg_rating", "predicted_rating", "url", "genre_name"]]
        recs.rename(columns={"avg_rating":"average","predicted_rating":"predicted","genre_name":"genre"}, inplace=True)
        
        # Create a dictionary that includes the order of columns
        result_dict = {
            'columns': list(recs.columns),
            'data': recs.to_dict(orient='records')
        }
        
        # Return the result to the front-end
        return jsonify(result_dict)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)