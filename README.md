# Projects

### Shelf Life: Goodreads Book Recommender

**[Try the live app →](https://goodrec-137926939938.us-central1.run.app)** (no Goodreads account needed: click *See a demo*)

**Description:**
A web app that recommends books from your Goodreads history. Upload your Goodreads library export (or search and rate a few books), and it ranks ~105,000 books for you, predicts the stars you'd give each one, and explains every pick ("because you liked…"). Built on the UCSD Book Graph: 15.7M ratings from 465k readers.

* **Hybrid recommender:** item-item similarity blended with an implicit-feedback ALS taste model, folded in for each new user at request time. The blend shifts from one to the other as you rate more books.
* **Measured offline:** tuned on 9,906 held-out readers no model saw (NDCG@20 of 0.139 with a full history, vs. 0.036 for "most popular").
* **Personal predicted ratings:** calibrated to your own rating scale, stretched only as far as your ratings support.
* **Assistant tab:** a Claude-powered reading assistant (Haiku for quick questions, Sonnet for harder ones) that calls the recommender as tools, finds post-2017 books via web search, and talks books like a book club.
* **Engineering:** an offline pipeline (11 stages) produces the models; a FastAPI + React/TypeScript app serves recommendations in tens of milliseconds, deployed on Google Cloud Run. It replaces the 2023 version below, which took about a minute per request.

![Shelf Life homepage](01%20-%20Shelf%20Life%20%28Goodreads%20Recommender%20v2%29/docs/screenshot-home.png)

**Filetree:**
```
├── config/          pipeline settings, genre map, homepage picks
├── src/goodrec/
│   ├── pipeline/    s00_download … s10_package (offline)
│   ├── core/        scoring, similar readers, predictions (shared by the API and eval)
│   ├── api/         FastAPI app, CSV import matching, Assistant (chat)
│   └── eval/        offline evaluation
├── frontend/        React + Vite + TypeScript
└── tests/
```
See the [project README](01%20-%20Shelf%20Life%20%28Goodreads%20Recommender%20v2%29/README.md) for details.

___

### The TrashBot Project: Using drones and computer vision to find, map, and clean unregulated dumpsites in The Gambia

Full Report: https://drive.google.com/file/d/1KwQvrzWQVAILF3BFSnOUfve1DKAJXGcg/view?usp=sharing

**Description:**
This project, completed as a master's thesis, uses drone orthomapping and computer vision to assist Kanifing Municipality in finding and measuring unregulated dumpsites so that they can be cleaned and regulated before causing harm to residents' health and the environment. 

<img width="1229" alt="Screenshot 2023-08-07 at 12 20 06 PM" src="https://github.com/mraottth/TrashBot/assets/64610726/907ce1ab-54a7-47b9-9e2e-6ae2eb73f3ee">

 
<img width="1026" alt="Screenshot 2023-08-07 at 12 26 10 PM" src="https://github.com/mraottth/TrashBot/assets/64610726/a593f925-a390-4c49-8290-a93eff717181">

**Filetree:**
```
├── train_trashbot.ipynb
└── trashbot_predict.py
```

___

### Goodreads Book Recommender (2023 original)

*Superseded by [Shelf Life](#shelf-life-goodreads-book-recommender), the 2026 rebuild above.*


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

### Health Inspection Predictor

**Description:**
Uses [public data](https://data.cityofnewyork.us/Transportation/Open-Restaurants-Inspections/4dx7-axux) on restaurant health inspections in New York City to predict the score a restaurant will receive on its next inspection.

<img width="1414" alt="Screenshot 2023-08-10 at 7 36 09 PM" src="https://github.com/mraottth/projects/assets/64610726/9af8dddc-0dbc-4dc3-b7e0-92ef64231c12">


**Filetree:**
```
├── NYC Restaurant Inspection ML.pdf
└── Predicting_Restaurant_Inspections.ipynb
```

___

### COVID Visualizations

**Description:**
These data visualizations were created for the City of Seattle's vaccine distribution task force with the goal of 
making it easy to visualize the following insights in one simple view:

1. What is the current state of COVID cases and deaths in Seattle and other cities?
2. How does the current state of the pandemic compare to recent months?
3. How does the current state of the pandemic compare to all prior months?


Cases                      |  Deaths
:-------------------------:|:-------------------------:
![cases_pandemic_history_drilldown](https://github.com/mraottth/projects/assets/64610726/950fce3f-3ecc-4f0b-a3cb-57b1ae354fa4) | ![deaths_pandemic_history_drilldown](https://github.com/mraottth/projects/assets/64610726/a5434bd8-25d5-4cd5-ac06-45876f562929)


**Filetree:**
```
├── Figures
│   └── Cases
│       └── ...
│   └── Deaths
│       └── ...
├── cases_interactive.py
├── full_pandemic_history_cases_drilldown.py
└── full_pandemic_history_deaths_drilldown.py
```

___

### ADU Code Enforcement

**Description:**
Completed as part of the Stanford RegLab's analysis on whether municipal code enforcement of 
unpermitted accessory dwelling unit (ADU) construction disproportionately targets 
disadvantaged communities 

<img width="1412" alt="Screenshot 2023-08-10 at 8 07 10 PM" src="https://github.com/mraottth/projects/assets/64610726/94f48e25-fe59-43ae-9f17-f13442398fbd">


**Filetree:**
```
├── Census_Blocks_2020
│   ├── Census_Blocks_2020.dbf
|   ├── Census_Blocks_2020.shx
│   └── Census_Blocks_2020.xml
├── Census_Tracts_2020
│   ├── Census_Tracts_2020.cpg
|   ├── Census_Tracts_2020.dbf
|   ├── Census_Tracts_2020.prj
|   ├── Census_Tracts_2020.shx
│   └── Census_Tracts_2020.xml
└── LA_ADU_EDA_V2.ipynb
```

___

### OSCAR LDA

**Description:**
Topic modeling for an NLP project using BERT to summarize clinical articles. Full project [here](https://github.com/mlkimmins/OSCAR/tree/master)

<img width="921" alt="Screenshot 2023-08-10 at 6 26 54 PM" src="https://github.com/mraottth/projects/assets/64610726/a0641dae-b539-4071-82c5-c6e442d980bc">


**Filetree:**
```
└── topic_modeling.ipynb
```

___

### MD Unemployment Insurance Analysis

**Description:**
Explores and forecasts Maryland unemployment insurance activity from July 2008 to April 2013 ([Maryland Open Data Portal](https://opendata.maryland.gov/Business-and-Economy/Unemployment-Insurance-Data-July-2008-to-April-201/3x6e-7i3k/about_data)): new claims, people drawing benefits, dollars paid, and first vs. final checks issued. Seasonal decomposition and auto-ARIMA models (pmdarima) forecast each series.

**Filetree:**
```
└── md-ds-roth.ipynb
```

___

### Caixin Scraper

**Description:**
This web scraper was written to assist a research project at Harvard's Belfer Center seeking to identify 
cases of corruption in China that appear in the media before being officially announced by the 
Central Commission for Discipline Inspection (typically, it is the other way around in China).

**Filetree:**
```
├── data
│   ├── CCDI_Selected_Data.csv
|   ├── keywords.csv
│   └── scraped_results_0226.csv
└── caixin_webscraper.py
```

___

### Earthquake

**Description:**
Entry to DrivenData's competition, [Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/), which tasks participants with creating a model to 
predict the level of damage to buildings caused by the 2015 Nepal earthquake. Scored in top 2%.

**Filetree:**
```
├── Data
│   ├── test_values.csv
|   ├── train_labels.csv
│   └── train_values.csv
└── earthquake_model.py
```
