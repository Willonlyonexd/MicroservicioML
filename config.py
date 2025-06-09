import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# Cargar variables de entorno desde .env si existe
load_dotenv()

class DatabaseConfig(BaseModel):
    """Configuración de bases de datos"""
    # MongoDB
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb+srv://houwenvt:will@cluster0.crz8eun.mongodb.net/")
    mongo_db: str = os.getenv("MONGO_DB", "microservicio_ml")
    
    # PostgreSQL (ERP)
    postgres_uri: str = os.getenv("POSTGRES_URI", "postgresql://postgres:will@localhost:5432/erp_zamo1")

class ETLConfig(BaseModel):
    """Configuración del proceso ETL"""
    # Tablas a sincronizar - Actualizadas para incluir todas las tablas necesarias
    tables_to_sync: List[str] = [
        "pedido", 
        "pedido_detalle", 
        "producto", 
        "cliente",
        "mesa",
        "categoria",
        "usuario",
        "cuenta_mesa",  # Añadido para análisis temporal
        "reserva",      # Añadido para análisis de comportamiento
        "venta"         # Añadido para confirmar pedidos completados
    ]
    
    # Intervalos de sincronización (en minutos) - Actualizado a 3 minutos
    sync_interval: int = int(os.getenv("SYNC_INTERVAL", "3"))
    
    # Tamaño de batch para procesamiento - Ahora opcional en extract_table
    # Si se pasa como None, se extraen todos los registros sin límite
    batch_size: int = int(os.getenv("BATCH_SIZE", "30000"))

class MLConfig(BaseModel):
    """Configuración de modelos ML"""
    # Directorio para almacenar modelos
    models_dir: str = os.getenv("MODELS_DIR", "./models")
    
    # Configuración de pronóstico
    forecast_window: int = int(os.getenv("FORECAST_WINDOW", "14"))  # días a predecir
    forecast_features: List[str] = ["dia_semana", "mes", "dia_mes", "festivo", "ventas_previas"]
    
    # Configuración de segmentación
    segmentation_max_clusters: int = int(os.getenv("MAX_CLUSTERS", "8"))

class APIConfig(BaseModel):
    """Configuración de la API"""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    enable_playground: bool = os.getenv("ENABLE_PLAYGROUND", "True").lower() == "true"

class LogConfig(BaseModel):
    """Configuración de logging"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    file: Optional[str] = os.getenv("LOG_FILE")

class Config(BaseModel):
    """Configuración global de la aplicación"""
    app_name: str = "Restaurant ML Microservice"
    version: str = "0.1.0"
    
    # Componentes
    database: DatabaseConfig = DatabaseConfig()
    etl: ETLConfig = ETLConfig()
    ml: MLConfig = MLConfig()
    api: APIConfig = APIConfig()
    log: LogConfig = LogConfig()
    
    # Si estamos en modo desarrollo
    dev_mode: bool = os.getenv("DEV_MODE", "False").lower() == "true"

# Crear instancia global de configuración
config = Config()

# Función para obtener configuración
def get_config() -> Config:
    return config

if __name__ == "__main__":
    # Imprimir configuración si se ejecuta directamente
    import json
    print(json.dumps(config.model_dump(), indent=2))