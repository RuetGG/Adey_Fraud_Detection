import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

def remove_duplicates(df):
    return df.drop_duplicates()

def missing_values(df):
    return df.isnull().sum()
def convert_to_datetime(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df
