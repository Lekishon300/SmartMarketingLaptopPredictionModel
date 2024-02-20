# project: p5
# submitter: lekishon
# partner: none
# hours: 5

import pandas as pd
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.compose
import sklearn.linear_model
from sklearn.model_selection import cross_val_score
import numpy as np

class UserPredictor:
    def add_columns(self, users, logs):
        users = users.set_index('user_id')
        laptop_seconds = logs[logs['url'].str.contains('laptop')].groupby('user_id')['seconds'].sum() #time for laptop site
        t_seconds = logs.groupby('user_id')['seconds'].sum() #time by user

        users['t_seconds'] = t_seconds #new columns
        users['laptop_mins'] = laptop_seconds/60
        
        users.fillna({'t_seconds': 0, 'laptop_mins':0}, inplace=True) # Fill NaN values with 0

        users.reset_index(inplace=True) 
        return users

    def fit(self, train_users, train_logs, train_y):
       
        train_users_new = self.add_columns(train_users, train_logs)
        
        cont = ["past_purchase_amt", "t_seconds","laptop_mins"]
        imput_c = sklearn.impute.SimpleImputer(strategy="median")
        transform_c = sklearn.preprocessing.StandardScaler()
        steps_c = sklearn.pipeline.Pipeline(steps=[("ic", imput_c), ("tc", transform_c)])
        
        discrete = ["badge"]
        imput_d = sklearn.impute.SimpleImputer(strategy = "constant", fill_value = "None")
        transform_d = sklearn.preprocessing.OneHotEncoder()
        steps_d = sklearn.pipeline.Pipeline(steps = [("id", imput_d), ("td", transform_d)])
        
        pre = sklearn.compose.ColumnTransformer(transformers=[("c", steps_c, cont),("d", steps_d, discrete)])
        self.model = sklearn.pipeline.Pipeline(steps=[("pre", pre), ("clf", sklearn.linear_model.LogisticRegression())])
        
        X = train_users_new[cont + discrete]
        y = train_y['y']
        
        self.model.fit(X, y)
    
    def predict(self, test_users, test_logs):
        test_users_new = self.add_columns(test_users, test_logs)
        X_test = test_users_new[["past_purchase_amt", "t_seconds","laptop_mins","badge"]]
        
        return self.model.predict(X_test)

        
        
        
    
