# Projects

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
├── Cases
│   ├── ...
│   └── ...
├── Deaths
│   ├── ...
│   └── ...
├── cases_interactive.py
├── full_pandemic_history_cases_drilldown.py
└── full_pandemic_history_deaths_drilldown.py
```
___

### Caixin Scraper

**Description:**
This web scraper was written to assist a research project at Harvard's Belfer Center seeking to identify 
cases of corruption in China that appear in the media before being officially announced by the 
Central Commission for Discipline Inspection (typically, it is the other way around in China).

**Filetree:**
```
├── Data
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

___

### Goodreads Book Recommender

**Description:**
Uses [goodreads data](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0) scraped by Mengting Wan and Julian McAuley at UCSD to build a recommender system using collaborative filtering via KNN to suggest books and allow for filtering on genre, rating, and other features.

* **00_prep_goodreads_data.ipynb** imports, cleans, and prepares the UCSD data for later steps
* **01_infer_genres.ipynb** performs topic modeling via Latent Dirichlet Allocation (LDA) to infer genres based on each book's description text. These genres are used for making recommendations in the next step
* **02_book_recommender.ipynb** generates book recommendations with user-user similarity via KNN and user-item rating predictions via matrix factorization

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

### OSCAR LDA

**Description:**
Topic modeling for an NLP project using BERT to summarize clinical articles. Full project [here](https://github.com/mlkimmins/OSCAR/tree/master)

<img width="921" alt="Screenshot 2023-08-10 at 6 26 54 PM" src="https://github.com/mraottth/projects/assets/64610726/a0641dae-b539-4071-82c5-c6e442d980bc">


**Filetree:**
```
└── topic_modeling.ipynb
```
