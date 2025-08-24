import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Modelo regresión logística.
from sklearn.neighbors import KNeighborsClassifier  # Modelo KNN.
from sklearn.tree import DecisionTreeClassifier  # Árbol de decisión.
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from limpieza import clean 

clean()
df = pd.read_csv('/app/bases_modelo/base_penguin.csv')
df = pd.DataFrame(df)
X = df.drop('species', axis=1)
Y = df['species']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, y_pred))
joblib.dump(model, '/models/KNeighborsClassifier.pkl')
print('Modelo Exportado Exitosamente')