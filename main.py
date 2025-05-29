from config import get_config
from utils import logger, setup_logging
from db.mongo_client import get_mongo_manager
from etl.extract import get_postgres_extractor
from etl.sync import get_data_synchronizer
from db.models import create_indexes
import time

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
    """Prueba la sincronización de datos para todas las tablas configuradas.
    
    Args:
        tenant_id: ID del tenant para sincronizar
        force_full: Si es True, fuerza una sincronización completa
        
    Returns:
        bool: True si todas las sincronizaciones fueron exitosas
    """
    logger.info(f"Probando sincronización para tenant {tenant_id}...")
    
    config = get_config()
    synchronizer = get_data_synchronizer()
    
    all_success = True
    results = {}
    
    # Sincronizar todas las tablas configuradas
    for table_name in config.etl.tables_to_sync:
        try:
            logger.info(f"Sincronizando tabla: {table_name}...")
            start_time = time.time()
            
            # Modificado para usar force_full como parámetro
            result = synchronizer.sync_table(tenant_id=tenant_id, table_name=table_name, force_full=force_full)
            
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            if result.get("status") == "success":
                logger.info(f"Sincronización de {table_name} completada en {duration}s: {result.get('records', 0)} registros")
            else:
                logger.warning(f"Sincronización de {table_name} completada con estado: {result.get('status')}")
                if result.get("status") == "error":
                    all_success = False
            
            results[table_name] = result
            
        except Exception as e:
            logger.error(f"Error sincronizando {table_name}: {str(e)}")
            all_success = False
            results[table_name] = {"status": "error", "error": str(e)}
    
    # Mostrar resumen
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    error_count = sum(1 for r in results.values() if r.get("status") == "error")
    skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
    
    logger.info(f"Resumen de sincronización: {success_count} exitosas, {error_count} errores, {skipped_count} omitidas")
    
    return all_success

def sync_scheduled(tenant_id=1, interval_minutes=None, force_initial_full=False):
    """Ejecuta la sincronización de forma programada.
    
    Args:
        tenant_id: ID del tenant para sincronizar
        interval_minutes: Intervalo en minutos entre sincronizaciones
        force_initial_full: Si es True, la primera sincronización será completa
    """
    config = get_config()
    
    if interval_minutes is None:
        interval_minutes = config.etl.sync_interval
    
    logger.info(f"Iniciando sincronización programada cada {interval_minutes} minutos para tenant {tenant_id}")
    
    synchronizer = get_data_synchronizer()
    first_run = True
    
    try:
        while True:
            logger.info("Ejecutando sincronización programada...")
            start_time = time.time()
            
            try:
                # Determinar si esta ejecución debe ser completa o incremental
                use_force_full = first_run and force_initial_full
                if use_force_full:
                    logger.info("Realizando sincronización completa inicial...")
                else:
                    logger.info("Realizando sincronización incremental...")
                
                # Ejecutar sincronización con el modo apropiado
                results = synchronizer.sync_all_tables(
                    tenant_id=tenant_id, 
                    force_full=use_force_full
                )
                
                first_run = False  # Ya no es la primera ejecución
                
                end_time = time.time()
                duration = round(end_time - start_time, 2)
                
                # Calcular estadísticas
                success_count = sum(1 for r in results if r.get("status") == "success")
                error_count = sum(1 for r in results if r.get("status") == "error")
                skipped_count = sum(1 for r in results if r.get("status") == "skipped")
                records_count = sum(r.get("records", 0) for r in results if r.get("status") == "success")
                
                logger.info(f"Sincronización completada en {duration}s: {success_count} tablas exitosas, {error_count} errores, {skipped_count} omitidas, {records_count} registros procesados")
            except Exception as e:
                logger.error(f"Error durante el ciclo de sincronización: {str(e)}")
                # Continuar con el siguiente ciclo aunque este haya fallado
            
            # Esperar hasta la próxima ejecución
            next_sync_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + interval_minutes * 15))
            logger.info(f"Próxima sincronización programada a las {next_sync_time} (en {interval_minutes} minutos)")
            time.sleep(interval_minutes * 15)
            
    except KeyboardInterrupt:
        logger.info("Sincronización programada interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error fatal en sincronización programada: {str(e)}")
        raise  # Re-lanzar para permitir que main() maneje la excepción

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
    
    # Decidir si forzar sincronización completa inicial en modo desarrollo
    force_initial_full = config.dev_mode
    
    # Siempre ejecutar la sincronización programada, independientemente del modo
    try:
        # CÓDIGO DE SINCRONIZACIÓN COMENTADO - INICIO
        # En modo desarrollo, podemos ejecutar una sincronización inicial completa
        # En producción, vamos directamente a la sincronización programada incremental
        # if config.dev_mode:
        #     logger.info("Modo desarrollo: Ejecutando sincronización inicial completa...")
        #     test_result = test_sync(tenant_id=1, force_full=True)
        #     
        #     if not test_result:
        #         logger.warning("La sincronización inicial tuvo errores. Verificar los logs antes de continuar.")
        # 
        # # Iniciar la sincronización programada 24/7
        # logger.info("Iniciando servicio de sincronización 24/7...")
        # # En desarrollo, la primera sincronización programada será incremental
        # # después de la completa inicial que ya hicimos
        # sync_scheduled(tenant_id=1, force_initial_full=False)
        # CÓDIGO DE SINCRONIZACIÓN COMENTADO - FIN
        
        # INSERTA TU CÓDIGO ML AQUÍ
        logger.info("Sincronización desactivada. Usando datos existentes en MongoDB para ML.")
        
        # Ejemplo: código para trabajar con MongoDB directamente
        # mongo = get_mongo_manager()
        # data = list(mongo.db.raw_pedidos.find({"tenant_id": 1}))
        # logger.info(f"Procesando {len(data)} pedidos para modelo ML")
        
        # Agregar tiempo de espera para que el programa no termine inmediatamente
        logger.info("Presiona Ctrl+C para terminar")
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por el usuario")
    except Exception as e:
        logger.critical(f"Error fatal en el servicio: {str(e)}")
    finally:
        logger.info("Servicio finalizado. Cerrando conexiones...")
        # Cerrar conexiones explícitamente
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