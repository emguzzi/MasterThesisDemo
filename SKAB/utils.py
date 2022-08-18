import os
import pandas as pd


def read_SKAB(anomaly_free = False):
# open the .csv files from the SKAB datasets and return a list of pandas df
# anomaly_free = True if we also want to load the anomaly-free run. The df is then
# converted to numpy and returned separately
  
    all_files=[]
    for root, dirs, files in os.walk("../Data/SKAB_data/"):
        for file in files:
            if file.endswith(".csv"):
                 all_files.append(os.path.join(root, file))

    list_of_df = [pd.read_csv(file, 
                              sep=';', 
                              parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    if anomaly_free:
        path_anomaly_free = 'data/anomaly-free/anomaly-free.csv'
        df_anomaly_free = pd.read_csv(path_anomaly_free,sep = ';')
        df_anomaly_free = df_anomaly_free.drop(columns = ['datetime'])
        paths_anomaly_free = df_anomaly_free.to_numpy()
        return list_of_df, paths_anomaly_free
    else:
        return list_of_df