import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
import xgboost as xgb

def load_split(df, target_col='Class', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    return X_train, y_train, X_test, y_test

def get_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
def get_xgboost(scale_pos_weight):
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state = 42,
        use_label_encoder= False,
        eval_metric='logloss'
    )
    
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)
    return {'f1 ': f1, 'auc_pr ': auc_pr, 'confusion matrix ': cm}

def cross_validate_model(model_class, X, y, n_splits=5, **model_params):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores, auc_pr_scores = [], []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = model_class(**model_params)
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:,1]
        
        f1_scores.append(f1_score(y_val, y_val_pred))
        precision, recall, _ = precision_recall_curve(y_val, y_val_prob)
        auc_pr_scores.append(auc(recall, precision))
        
        return {
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'auc_pr_mean': np.mean(auc_pr_scores),
            'auc_pr_std': np.std(auc_pr_scores)
        }