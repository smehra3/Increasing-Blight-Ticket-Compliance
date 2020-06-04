
# Predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def blight_model():
    
    # Your code here
    train_df = pd.read_csv('train.csv', sep=',', engine='python')
    test_df = pd.read_csv('test.csv', sep=',', engine='python')
    #train_data = np.genfromtxt('readonly/train.csv', delimiter=",", skip_header=1, usecols=np.arange(0,35));
    #print(train_data[0])
    train_df = train_df[np.isfinite(train_df['compliance'])]
    
    feature_cols = ['agency_name','violation_zip_code',
                 'zip_code','ticket_issued_date','hearing_date','violation_code','disposition',
                 'fine_amount','judgment_amount','discount_amount']
    drop_cols = [x for x in train_df.columns if x not in feature_cols]
    #train_df = train_df.drop(drop_cols, axis=1, inplace=True)
    
    train_df.drop(['inspector_name', 'violator_name', 'non_us_str_code', 'violation_description', 
                'grafitti_status', 'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date',
                'payment_amount', 'balance_due', 'payment_date', 'payment_status', 
                'collection_status', 'compliance_detail', 
                'violation_zip_code', 'country','violation_street_number',
                'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 
                'city', 'state', 'zip_code'], axis=1, inplace=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['disposition'].append(test_df['disposition'], ignore_index=True))
    train_df['disposition'] = label_encoder.transform(train_df['disposition'])
    test_df['disposition'] = label_encoder.transform(test_df['disposition'])
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['violation_code'].append(test_df['violation_code'], ignore_index=True))
    train_df['violation_code'] = label_encoder.transform(train_df['violation_code'])
    test_df['violation_code'] = label_encoder.transform(test_df['violation_code'])
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['agency_name'].append(test_df['agency_name'], ignore_index=True))
    train_df['agency_name'] = label_encoder.transform(train_df['agency_name'])
    test_df['agency_name'] = label_encoder.transform(test_df['agency_name'])
    train_columns = list(train_df.columns.values)
    print(train_columns)
    train_columns.remove('compliance')
    test_df = test_df[train_columns]
    
    X = train_df.ix[:, train_df.columns != 'compliance']
    y = train_df['compliance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    #print(type(y_test))
    gb = GradientBoostingRegressor(learning_rate=0.01, max_depth=5, random_state=0).fit(X_train, y_train)
    #grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 10]}
    #grid_clf_auc = GridSearchCV(gb, param_grid=grid_values, scoring='roc_auc')
    #grid_clf_auc.fit(X_train, y_train)
    #print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc.best_score_)
    roc_auc = roc_auc_score(y_test, gb.predict(X_test))

    #print(roc_auc)
    return pd.DataFrame(gb.predict(test_df.dropna()), test_df.ticket_id)# Your answer here


# In[49]:

result_df=blight_model()


# In[50]:

print(result_df)


# In[ ]:



