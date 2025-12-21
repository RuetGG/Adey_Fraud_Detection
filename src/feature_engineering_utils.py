import ipaddress
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except Exception:
        return None

def convert_ip_column(df, column):
    df[column] = df[column].apply(ip_to_int)
    df[column] = df[column].astype('Int64')
    return df


def add_time_features(df, time_col):
    df['hour_of_day'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    return df

def time_since_signup(df, signup_col, purchase_col):
    df['time_since_signup'] = (
        df[purchase_col] - df[signup_col]
    ).dt.total_seconds()/3600
    return df

def apply_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)

def add_transaction_counts(df, user_col='user_id'):
    txn_counts = df.groupby(user_col)['purchase_time'].count().reset_index()
    txn_counts.rename(columns={'purchase_time': 'txn_count'}, inplace=True)
    df = df.merge(txn_counts, on=user_col, how='left')
    return df

def scale_numerical(df, num_cols):
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def encode_categorical(df, cat_cols):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df