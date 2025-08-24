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
