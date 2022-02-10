import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning) # Turn off futurewarnings

# Read in datasets
filepath = '/Users/mattroth/Documents/DS Competitions/Earthquake/'

test_vals = pd.read_csv(filepath + 'test_values.csv', index_col='building_id')
train_vals = pd.read_csv(filepath + 'train_values.csv', index_col='building_id').sort_index()
train_labels = pd.read_csv(filepath + 'train_labels.csv', index_col='building_id').sort_index()
submission_format = pd.read_csv(filepath + 'submission_format.csv')
submission_index = pd.DataFrame(test_vals.index) # Save order for submission file
test_vals = test_vals.sort_index()
# Combine test and train data into one dataframe
train_test_concat = pd.concat([train_vals, test_vals])

# Define function to cap values for features
def cap_val(df, col=[], val=[]):
    for c, v in zip(col, val):
        df[c].values[df[c].values > v] = v
# Call function on train & test vals
cap_cols = ['age','height_percentage', 'area_percentage']
cap_vals = [42, 16, 70]
cap_val(train_test_concat, col=cap_cols, val=cap_vals)


# Target and Count Encoding for geo
import category_encoders as ce
labels_vals = pd.merge(train_vals[['geo_level_1_id','geo_level_2_id', 'geo_level_3_id']], train_labels, left_index=True, right_index=True)

# Count Encode
for col in ['geo_level_1_id','geo_level_2_id', 'geo_level_3_id']:
    co = ce.CountEncoder(cols=[col])
    co.fit(labels_vals[col])
    train_test_concat[col+'_count'] = co.transform(train_test_concat[col])  

# Target Encoding for geo
for col in ['geo_level_1_id','geo_level_2_id', 'geo_level_3_id']:
    te = ce.TargetEncoder(cols=[col])
    te.fit(labels_vals[col], labels_vals['damage_grade'])
    train_test_concat[col] = te.transform(train_test_concat[col])

# Add height to area ratio feature
train_test_concat['height_to_area'] = train_test_concat['height_percentage'] / train_test_concat['area_percentage']
train_test_concat['fam_per_fl'] = train_test_concat['count_families'] / train_test_concat['count_floors_pre_eq']
train_test_concat['total_geo_damage1'] = train_test_concat['geo_level_1_id'] * train_test_concat['geo_level_1_id_count']
train_test_concat['total_geo_damage2'] = train_test_concat['geo_level_2_id'] * train_test_concat['geo_level_2_id_count']
train_test_concat['total_geo_damage3'] = train_test_concat['geo_level_3_id'] * train_test_concat['geo_level_3_id_count']

for col in ['geo_level_1_id_count', 'geo_level_2_id_count', 'geo_level_3_id_count', 'total_geo_damage1', 'total_geo_damage2', 'total_geo_damage3']:
    train_test_concat[col] = train_test_concat[col].fillna(train_test_concat[col].mean())

# Separate out labels by dtype - separate dataframe and set data types
cont = [
        # 'geo_damage',
        'age',
        'height_percentage',
        'area_percentage',
        'count_families',
        # 'height_damage',
        'geo_level_1_id',
        'geo_level_2_id',
        'geo_level_3_id',
        # 'geo_composite',
        # 'age_damage',
        'height_to_area',
        'count_floors_pre_eq',
        'fam_per_fl',
        # 'geo_level_1_id_count',
        # 'geo_level_2_id_count',
        # 'geo_level_3_id_count',
        'total_geo_damage1',
        'total_geo_damage2',
        'total_geo_damage3'
        ]

cat_multi = [
    'foundation_type',
    'roof_type',
    'ground_floor_type',
    'other_floor_type',
    'position',
    'land_surface_condition',
    'plan_configuration',
    'legal_ownership_status'
    # 'geo_level_2_id'
    # 'height_percentage_category'
    ]

cat_binary = [x for x in train_vals.columns if x.startswith('has')]


train_test_concat_cont = train_test_concat[cont]
train_test_concat_cat_multi = train_test_concat[cat_multi]
train_test_concat_cat_binary = train_test_concat[cat_binary]
# Create new column to sum binary columns
train_test_concat_cont['bin_sum'] = train_test_concat_cat_binary.apply(np.sum, axis=1)
# Drop unnecessary columns
train_test_concat_cat_binary.drop(['has_secondary_use_rental', 'has_superstructure_stone_flag', 'has_secondary_use_other'], axis=1, inplace=True)

cat_binary = [
    'has_superstructure_mud_mortar_stone',
    'has_superstructure_cement_mortar_brick',
    'has_superstructure_mud_mortar_brick',
    'has_superstructure_rc_non_engineered',
    'has_superstructure_rc_engineered',
    'has_secondary_use',
    'has_secondary_use_hotel',
    'has_superstructure_adobe_mud',
    'has_superstructure_bamboo',
    'has_superstructure_timber',
    'has_superstructure_stone_flag',
    'has_secondary_use_agriculture',
    'has_secondary_use_other'
    ]

train_test_concat_cat_binary = train_test_concat[cat_binary]

# One hot encode multi-level categorical variables
train_test_concat_cat_multi = pd.get_dummies(train_test_concat_cat_multi, columns=cat_multi, drop_first=True)

# Label Encode multi-level categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in train_test_concat_cat_multi.columns:
    le.fit(train_test_concat_cat_multi[train_test_concat_cat_multi.index.isin(train_vals.index)][col])
    train_test_concat_cat_multi[col] = le.transform(train_test_concat_cat_multi[col])

# # Scale and PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# ss = StandardScaler()
# pca = PCA()
# for m in [ss, pca]:
#     train_test_concat_cont = pd.DataFrame(
#                                       m.fit(train_test_concat_cont[train_test_concat_cont.index.isin(train_vals.index)])\
#                                       .transform(train_test_concat_cont),\
#                                       columns=train_test_concat_cont.columns, 
#                                       index=train_test_concat_cont.index
#                                       )

# Merge features back together
X_merge = pd.concat([train_test_concat_cont, train_test_concat_cat_multi, train_test_concat_cat_binary], axis=1)#.loc[:, ['geo_damage','age','height_percentage','area_percentage','bin_sum','has_superstructure_cement_mortar_brick']]
y = train_labels['damage_grade']

# Split out test and train from concat
X = X_merge[X_merge.index.isin(train_vals.index)] 
test_vals_X = X_merge[X_merge.index.isin(test_vals.index)] 


print('Ending Data Cleaning & Preprocessing...')
# END DATA CLEANING // PREPROCESSING
# START ML
print('Starting ML...')

# Split data into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Import ML Libararies
# from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb
# from sklearn.tree import DecisionTreeClassifier

# Instantiate Models
# lr = LogisticRegression(C=10, max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1) 
rf = RandomForestClassifier(n_estimators=150, max_depth=17, min_samples_split=2, n_jobs=-1) 
xgb_c = xgb.XGBClassifier(n_estimators=80, objective='multi:softmax', num_class=3, n_jobs=-1)
xgb_2 = xgb.XGBClassifier(n_estimators=500, objective='multi:softmax', num_class=3, n_jobs=-1)
vc = VotingClassifier(estimators=[('rf',rf), ('xgb',xgb_c), ('xgb_2',xgb_2)], n_jobs=-1, verbose=True, voting='hard', weights=[2,2,1])


vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
print(classification_report(y_test, y_pred))
print("vc:", round(f1_score(y_test, y_pred, average='micro'), 3))

rf.fit(X_train, y_train)
importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
sorted_importances_rf = importances_rf.sort_values()
fig, ax = plt.subplots()
fig.set_size_inches(8, 11.5)
sorted_importances_rf.plot(kind='barh')
plt.show()

# vc.fit(X_train, y_train)
y_pred = vc.predict(test_vals_X)
results = pd.DataFrame({'building_id':test_vals.index, 'damage_grade':y_pred})
results = pd.merge(submission_index, results, on='building_id') # Merge back to submission index to get in proper order
results.to_csv('results_stacked' + str(np.random.randint(0,50000)) + '.csv', index=False)   
