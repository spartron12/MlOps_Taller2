# Imagen base
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app
RUN mkdir /models

# Copiamos pyproject.toml y el lockfile de uv
COPY pyproject.toml uv.lock ./

# Instalamos uv (en vez de hatch)
RUN pip install --upgrade pip \
    && pip install uv

# Instalamos dependencias en el sistema (sin virtualenv)
RUN uv sync --frozen

# Copiamos el resto del código
COPY . .

# Exponemos el puerto de Uvicorn (ajusta según tu app)
EXPOSE 9999

# Comando para levantar tu API con Uvicorn usando uv
# Ajusta main:app según tu archivo FastAPI
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9999"]

