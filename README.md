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

___

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
___

### The TrashBot Project: Using drones and computer vision to find, map, and clean unregulated dumpsites in The Gambia

Full Report: https://drive.google.com/file/d/1KwQvrzWQVAILF3BFSnOUfve1DKAJXGcg/view?usp=sharing

### Executive Summary

**Background & Motivation:** 

Residents of Kanifing Municipality in The Gambia face a serious issue: their neighborhoods are filled with openly dumped trash. Much of the waste produced by households, businesses, and pedestrians ends up on the ground in communities instead of properly disposed of at the municipal dumpsite. This is especially true in areas with limited access to municipal waste management services. In Ebo Town and Tallinding, for example, some unregulated dumpsites have grown to cover hundreds of thousands of square feet. There is significant concern among citizens and elected officials about the consequences of these open dumps – trash left to sit in communities threatens the health and safety of residents and pollutes the environment.


**Problem Statement:** 

Under Mayor Talib Ahmed Bensouda, Kanifing Municipal Council (KMC) has made it its number one priority to clean trash from public spaces and improve waste management. This is no easy task, especially because KMC’s budgetary and infrastructural constraints limit its operational capacity. Achieving better results requires getting more with less and precision-targeting interventions to areas of need. That is why this project focuses on improving KMC’s operational efficiency in cleaning openly dumped trash from communities.


**Project Scope:** 

We aim to develop a scalable, repeatable, ethical, and cost-effective way to find, map, and measure unregulated dumpsites so KMC can clean public spaces with precision & economy.


**Methodology:** 

First, we conduct aerial surveys of Kanifing neighborhoods using a camera drone. Next, we train a neural network to identify trash in aerial photos and apply it to our drone images. We call this computer vision model TrashBot. Finally, we stitch together images that have been run through TrashBot to create interactive maps that highlight openly dumped trash, allowing KMC to find, map, and measure unregulated dumpsites.


<img width="1229" alt="Screenshot 2023-08-07 at 12 20 06 PM" src="https://github.com/mraottth/TrashBot/assets/64610726/907ce1ab-54a7-47b9-9e2e-6ae2eb73f3ee">

**Results & Deliverables:** 

As a proof of concept, we surveyed over 600 acres in 5 Kanifing neighborhoods, where we located nearly 400,000 square feet of trash. We provide KMC with seven TrashBot maps and a suite of custom software tools so they can run the methodology independently.


<img width="1026" alt="Screenshot 2023-08-07 at 12 26 10 PM" src="https://github.com/mraottth/TrashBot/assets/64610726/a593f925-a390-4c49-8290-a93eff717181">

**Recommendations:** 

Due to privacy implications inherent to the use of drones and machine learning, we set technical & operational standards for the ethical use of our methodology. These include image obfuscation, data security measures, purpose limitation, and democratic controls.
Finally, we develop six use cases for KMC to apply our methods to its most pressing waste management issues. These include:
1. Intervention targeting
2. Optimizing receptacle placement
3. Waste management metric tracking
4. Finding and closing unregulated dumpsites 5. Modeling health and environmental risk
6. Expanding service range strategically

**Filetree:**
```
├── NYC Restaurant Inspection ML.pdf
└── Predicting_Restaurant_Inspections.ipynb
```
