import pandas as pd
from sklearn.metrics import r2_score
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from google.cloud import bigquery
from datetime import datetime
from google.cloud import storage
import os
import numpy as np


def main():
    client = bigquery.Client(project="sublime-state-413617")
    table_id = "model_metrics.bike_sharing_metrics"
    storage_client = storage.Client()
    bucket = storage_client.bucket('model-collections')
    data = client.query("""select * from sublime-state-413617.data.bike_sharing_day""").to_dataframe()
    columns = ['season', 'yr', 'holiday', 'atemp', 'casual', 'registered']
    
    data.drop(['instant', 'dteday'], axis=1, inplace=True)

    train_x, test_x, train_y, test_y = train_test_split(data[columns], data['cnt'], test_size=0.1)
    print('Model Training Started')
    rfc = RandomForestRegressor(max_depth=None,n_estimators=10)
    rfc.fit(train_x, train_y)
    predict = rfc.predict(test_x)
    r2 = r2_score(predict, test_y)
    print('Model Training done....')
    # checking if model is better performer or not

    
    pickle.dump(rfc, open('model.pkl', 'wb'))
    print('Model Saved!')
    bucket = storage_client.bucket('model-collections')
    blob = bucket.blob('model.pkl')
    blob.upload_from_filename('model.pkl')
    print('Model Pushed!')
    

    print('Training Completed')


if __name__=="__main__":
    main()