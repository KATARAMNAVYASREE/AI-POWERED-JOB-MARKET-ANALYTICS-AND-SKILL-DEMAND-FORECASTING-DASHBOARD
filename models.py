import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_role_model(data):

    pivot = pd.crosstab(data["job_title"], data["skills"])

    X = pivot.values
    y = pivot.index

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X,y_encoded)

    return model, pivot.columns, encoder

