FROM python:3.11-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Establecer directorio de trabajo
WORKDIR /app

# Copiar los archivos
COPY . /app

# Instalar librerías necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl git \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y gcc curl \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Exponer puerto de la API Flask
EXPOSE 5000

# Comando para iniciar el servidor Flask
CMD ["python", "server.py"]
