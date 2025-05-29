from config import get_config
from utils import logger, setup_logging
from db.mongo_client import get_mongo_manager
from etl.extract import get_postgres_extractor
from etl.sync import get_data_synchronizer
from db.models import create_indexes
import time

# AÑADIR ESTA VARIABLE AQUÍ PARA CONTROLAR LA SINCRONIZACIÓN
DISABLE_SYNC = True  # Cambia a False cuando quieras volver a activar la sincronización

def test_connections():
    """Prueba las conexiones a bases de datos."""
    logger.info("Probando conexiones a bases de datos...")
    
    # Probar MongoDB
    try:
        mongo = get_mongo_manager()
        # Evitar evaluar el objeto de base de datos como booleano
        mongo_info = mongo.client.server_info()
        logger.info(f"Conexión a MongoDB funciona correctamente: versión {mongo_info.get('version', 'desconocida')}")
    except Exception as e:
        logger.error(f"Error al conectar con MongoDB: {str(e)}")
        return False
    
    # Probar PostgreSQL
    try:
        postgres = get_postgres_extractor()
        result = postgres.execute_query("SELECT 1 as test")
        logger.info("Conexión a PostgreSQL funciona correctamente")
    except Exception as e:
        logger.error(f"Error al conectar con PostgreSQL: {str(e)}")
        return False
    
    return True

def test_sync(tenant_id=1, force_full=False):
    """Prueba la sincronización de datos para todas las tablas configuradas."""
    # Resto del código igual...

def sync_scheduled(tenant_id=1, interval_minutes=None, force_initial_full=False):
    """Ejecuta la sincronización de forma programada."""
    # Resto del código igual...

def main():
    """Función principal de la aplicación."""
    # Obtener configuración
    config = get_config()
    
    # Configurar logging
    setup_logging()
    
    # Mensaje de inicio
    logger.info(f"Iniciando {config.app_name} v{config.version}")
    logger.info(f"Modo desarrollo: {config.dev_mode}")
    
    # Probar conexiones
    if not test_connections():
        logger.error("Error en las conexiones a bases de datos. Abortando.")
        return
    
    # Crear índices en MongoDB
    try:
        create_indexes()
    except Exception as e:
        logger.error(f"Error al crear índices: {str(e)}")
    
    # MODIFICAR ESTA SECCIÓN PARA DESACTIVAR LA SINCRONIZACIÓN
    if not DISABLE_SYNC:
        # Código original de sincronización
        force_initial_full = config.dev_mode
        try:
            if config.dev_mode:
                logger.info("Modo desarrollo: Ejecutando sincronización inicial completa...")
                test_result = test_sync(tenant_id=1, force_full=True)
                
                if not test_result:
                    logger.warning("La sincronización inicial tuvo errores. Verificar los logs antes de continuar.")
            
            logger.info("Iniciando servicio de sincronización 24/7...")
            sync_scheduled(tenant_id=1, force_initial_full=False)
        except Exception as e:
            logger.critical(f"Error fatal en el servicio: {str(e)}")
    else:
        # AÑADIR AQUÍ TU CÓDIGO DE ML QUE USA LOS DATOS DE MONGODB
        logger.info("Sincronización desactivada. Usando datos existentes en MongoDB para ML.")
        
        # Ejemplo: Código para tu modelo de ML
        # from ml.model import train_model, predict
        # train_model()
        # predictions = predict(new_data)
        
        # Mantén esta parte para asegurar que las conexiones se cierran correctamente
        logger.info("Procesamiento ML completado.")
    
    # Cerrar conexiones en cualquier caso
    finally:
        logger.info("Servicio finalizado. Cerrando conexiones...")
        try:
            postgres = get_postgres_extractor()
            postgres.close()
        except:
            pass
        
        try:
            mongo = get_mongo_manager()
            mongo.close()
        except:
            pass
    
if __name__ == "__main__":
    main()