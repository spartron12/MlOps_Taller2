import palmerpenguins as pp
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
def clean():
    df = pp.load_penguins()
    df.head()
    df[df.isna().any(axis=1)]
    df.dropna(inplace=True)
    categorical_cols = ['sex','island']
    encoder = OneHotEncoder( handle_unknown='ignore')
    x = df.drop(columns=['species'])
    y = df['species']
    x_encoded = encoder.fit_transform(x[categorical_cols])
    X_numeric = x.drop(columns=categorical_cols)
    X_final = np.hstack((X_numeric.values, x_encoded.toarray()))
    joblib.dump(encoder, "encoder_variables.pkl")
    df_encoded = pd.get_dummies(df, columns=['island','sex'])
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    df_encoded.head()
    le = LabelEncoder()
    df_encoded['species'] = le.fit_transform(df_encoded['species'])
    df_encoded.to_csv('/app/bases_modelo/base_penguin.csv', index = False)
    print('Base exportada con éxito')
    return 