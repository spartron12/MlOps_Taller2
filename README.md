# MlOps_Taller2
Grupo compuesto por Sebastian Rodríguez y David Córdova
# Taller: Predicción de Especies de Pingüinos con FastAPI y Docker

Este proyecto abarca desde el entrenamiento de modelos de **Machine Learning** con Jupyter, hasta el despliegue de una API REST con **FastAPI** que permite hacer predicciones sobre nuevas observaciones de pingüinos.

## Estructura del Proyecto

La estructura del proyecto está organizada en dos partes principales: **Jupyter** para la preparación de modelos y **API** para el despliegue y uso de los modelos entrenados.

```
proyecto-pinguinos/
├── api/
│   ├── models/
│   ├── Dockerfile
│   ├── main.py
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
├── Jupyter/
│   ├── notebooks/
│   ├── crea_modelos.py
│   ├── limpieza.py
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
── docker-compose.yml
│  
── models/


### Descripción de Componentes

- **Jupyter**:
  - **crea_modelos.py**: Script para la creación y entrenamiento de modelos, se puede desplegar en un notebook de Jupyter si es necesario.
  - **limpieza.py**: Script para la limpieza y preparación de datos.
  - **notebooks/**: Carpeta con notebooks para exploración de datos y análisis.

- **API**:
  - **main.py**: Archivo principal para la implementación de la API con FastAPI.
  - **models/**: Carpeta con los modelos entrenados en formato pickle.
  - **Dockerfile**: Contenerización de la API.
  - **pyproject.toml**: Dependencias y configuración del proyecto.
  - **README.md**: Documentación de la API.

---

## 1. Entrenamiento de Modelos (Jupyter)

En esta sección se encuentran los scripts que realizan la **limpieza de datos** y el **entrenamiento de modelos** de **Machine Learning**.

### Limpieza de Datos
- **limpieza.py**: Realiza el preprocesamiento de los datos, como la eliminación de valores nulos y la codificación de variables categóricas.

### Creación de Modelos
- **crea_modelos.py**: Entrena modelos de clasificación (Regresión logística, Árbol de decisión y KNN) utilizando el conjunto de datos de **palmerpenguins**. Los modelos entrenados se guardan en la carpeta **models/** en formato **pickle**.

---

## 2. API REST con FastAPI

Una vez que los modelos han sido entrenados y guardados, se consume la API para realizar predicciones sobre nuevos datos.

### Funcionalidades de la API

La API permite:
- **Recepción de datos**: Mediante un esquema **Pydantic** para validar las entradas.
- **Selección dinámica de modelos**: Permite elegir entre diferentes modelos entrenados.
- **Inferencia escalada**: Los datos de entrada son automáticamente procesados.
- **Respuesta estructurada**: Devuelve la especie predicha junto con las probabilidades por clase.

---

## 3. Contenerización con Docker

Tanto la parte de la **API** como la de **Jupyter** están creadas en un contenedor con docker desplegado mediante compose  para mayor facilidad.

### Dockerfile para JupyterLab
El archivo **Dockerfile** en la carpeta **api/** define cómo se construye la imagen de Docker para la API FastAPI.

```dockerfile
# Imagen base ligera de Python
FROM python:3.12-slim

# Crear directorios
RUN mkdir -p /bases_modelo /encoder

# Copiar dependencias
COPY pyproject.toml uv.lock ./

# Instalar pip y uv
RUN pip install --upgrade pip \
    && pip install uv

# Instalar dependencias directamente en el sistema
RUN uv pip install -r pyproject.toml --system

# Directorio de trabajo
WORKDIR /app

# Copiar scripts
COPY crea_modelos.py .
COPY limpieza.py .

# Exponer puerto de Jupyter
EXPOSE 8888

# Arrancar Jupyter con Python del sistema
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token="]
```

### Dockerfile para API
El archivo **Dockerfile** en la carpeta **API/** define la imagen para que corra la aplicación que consume los modelos.
```dockerfile
# Imagen base
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app
RUN mkdir /models

# Copiamos pyproject.toml y el lockfile de uv
COPY pyproject.toml uv.lock ./

# Instalamos uv
RUN pip install --upgrade pip \
    && pip install uv

RUN uv sync --frozen

COPY . .

# Exponemos el puerto de Uvicorn 
EXPOSE 9999

# Comando para levantar tu API con Uvicorn usando uv
# Ajusta main:app según tu archivo FastAPI
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9999"]
```
---

## 4. Ejecución del Proyecto

1. **Levantamiento de la aplicación**:
```bash
# Imagen base
docker compose up
```
2. **Verificación de los modelos**:


![modelos_iniciales](./imagenes/modelos_iniciales.png)
*Figura 3: Resultado de la predicción mostrando la respuesta completa de la API*




### Entrenamiento de Modelos (Jupyter)
Para entrenar los modelos, sigue estos pasos:

1. **Activación del entorno de Jupyter**:


2. **Ejecución del script de limpieza de datos**:
 ```python
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

df_encoded = pd.get_dummies(df, columns=['island','sex'])
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
df_encoded.head()
df_encoded['species'] = df_encoded['species'].apply(lambda x: 
1 if x == 'Adelie' else 
2 if x == 'Chinstrap' else 
3 if x == 'Gentoo' else 
None)
df_encoded.to_csv('/bases_modelo/base_penguin.csv', index = False)
print('Base exportada con éxito')

   ```

3. **Entrenar modelos**:
   
 ```python
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

df = pd.read_csv('/bases_modelo/base_penguin.csv')
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

   ```

4. **Se guardan los modelos dentro del container para que la API los pueda consumir**:

### Ejecución de la API
Para ejecutar la API con Docker, sigue estos pasos:

1. **Construir y ejecutar con Docker Compose**:
   ```bash
   docker-compose up --build
   ```

La API estará disponible en http://localhost:9999/docs.

---

## 5. Endpoints Disponibles

- **GET** `/`: Página de bienvenida
- **POST** `/predict`: Endpoint de predicción
- **GET** `/docs`: Documentación interactiva (Swagger UI)
- **GET** `/redoc`: Documentación alternativa (ReDoc)
- **GET** `/health`: Endpoint de health check

### Ejemplo del funcionamiento 

-La API valida el contendor para ver que modelos ya han sido creados y pueden ser utilizados


---

## 6. Tecnologías Utilizadas

- **Machine Learning**: scikit-learn, pandas, numpy
- **API Framework**: FastAPI, Pydantic, Uvicorn
- **Contenerización**: Docker, Docker Compose
- **Data Source**: palmerpenguins dataset
- **Serialización**: pickle para persistencia de modelos
