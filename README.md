# MlOps_Taller2
Grupo compuesto por Sebastian Rodríguez y David Córdova
# Taller: Predicción de Especies de Pingüinos con FastAPI y Docker

Este proyecto abarca desde el entrenamiento de modelos de **Machine Learning** con Jupyter, hasta el despliegue de una API REST con **FastAPI** que permite hacer predicciones sobre nuevas observaciones de pingüinos.

## Estructura del Proyecto

La estructura del proyecto está organizada en dos partes principales: **Jupyter** para la preparación de modelos y **API** para el despliegue y uso de los modelos entrenados.

```
proyecto-pinguinos/
├── api/
│   ├── .venv/
│   ├── models/
│   ├── Dockerfile
│   ├── main.py
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
├── Jupyter/
│   ├── .venv/
│   ├── notebooks/
│   ├── crea_modelos.py
│   ├── limpieza.py
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
```

### Descripción de Componentes

- **Jupyter**:
  - **crea_modelos.py**: Script para la creación y entrenamiento de modelos.
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

### Dockerfile para Jupyter
El archivo **Dockerfile** en la carpeta **Jupyter/** define la imagen para los notebooks y la creación de los modelos.

---

## 4. Ejecución del Proyecto

### Entrenamiento de Modelos (Jupyter)
Para entrenar los modelos, sigue estos pasos:

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/DAVID316CORDOVA/Taller-1---MLOPS.git
   cd proyecto-pinguinos
   ```

2. **Crear entorno virtual y activar**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Linux/Mac
   .venv\Scripts\activate     # En Windows
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Entrenar modelos**:
   ```bash
   python Jupyter/crea_modelos.py
   ```

### Ejecución de la API
Para ejecutar la API con Docker, sigue estos pasos:

1. **Construir y ejecutar con Docker Compose**:
   ```bash
   docker-compose up --build
   ```

La API estará disponible en http://localhost:8989.

---

## 5. Endpoints Disponibles

- **GET** `/`: Página de bienvenida
- **POST** `/predict`: Endpoint de predicción
- **GET** `/docs`: Documentación interactiva (Swagger UI)
- **GET** `/redoc`: Documentación alternativa (ReDoc)
- **GET** `/health`: Endpoint de health check

### Ejemplo de Uso con cURL

```bash
curl -X POST "http://localhost:8989/predict?model_name=logistic_regression" \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "year": 2007,
    "sex_Female": 0,
    "sex_Male": 1,
    "island_Biscoe": 0,
    "island_Dream": 0,
    "island_Torgersen": 1
  }'
```

---

## 6. Tecnologías Utilizadas

- **Machine Learning**: scikit-learn, pandas, numpy
- **API Framework**: FastAPI, Pydantic, Uvicorn
- **Contenerización**: Docker, Docker Compose
- **Data Source**: palmerpenguins dataset
- **Serialización**: pickle para persistencia de modelos
