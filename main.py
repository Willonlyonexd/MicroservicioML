import pandas as pd
from datetime import datetime
from config import get_config
from utils import logger, setup_logging
from db.mongo_client import get_mongo_manager
from etl.extract import get_postgres_extractor
from etl.sync import get_data_synchronizer
from db.models import create_indexes
import time

# Importamos nuestro módulo de forecasting
from ml.forecast import TFForecaster
from ml.forecast.model import TFForecaster
from ml.forecast.config import MONGO_PARAMS, AGGREGATION_PARAMS
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import logging
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)



def diagnosticar_datos_disponibles():
    """Realiza un diagnóstico de los datos disponibles en MongoDB."""
    mongo = get_mongo_manager()
    
    logger.info("=== DIAGNÓSTICO DE DATOS EN MONGODB ===")
    
    # Listar todas las colecciones
    collections = mongo.db.list_collection_names()
    logger.info(f"Colecciones disponibles: {collections}")
    
    # Verificar datos en colecciones relevantes
    for collection_name in ["raw_pedidos", "raw_pedido_detalles", "raw_productos"]:
        if collection_name in collections:
            count = mongo.db[collection_name].count_documents({})
            logger.info(f"Colección {collection_name}: {count} documentos")
            
            # Mostrar un ejemplo de documento
            if count > 0:
                sample = mongo.db[collection_name].find_one({})
                logger.info(f"Ejemplo de {collection_name}: {list(sample.keys())}")
    
    # Verificar si hay datos de productos
    if "raw_productos" in collections:
        productos = list(mongo.db.raw_productos.find({}).limit(5))
        if productos:
            logger.info(f"Ejemplos de productos disponibles: {[p.get('_id') for p in productos]}")
        else:
            logger.warning("No hay productos en la colección raw_productos")
            
    # Verificar pedidos y detalles de pedidos
    if "raw_pedidos" in collections and "raw_pedido_detalles" in collections:
        # Obtener un pedido de ejemplo
        sample_pedido = mongo.db.raw_pedidos.find_one({})
        if sample_pedido:
            pedido_id = sample_pedido.get('_id')
            # Buscar detalles asociados a este pedido
            detalles = list(mongo.db.raw_pedido_detalles.find({"pedido_id": pedido_id}))
            logger.info(f"Pedido {pedido_id} tiene {len(detalles)} detalles")
            
            # Mostrar productos en esos detalles
            if detalles:
                productos_en_detalles = [d.get('producto_id') for d in detalles if 'producto_id' in d]
                logger.info(f"Productos en detalles de pedido: {productos_en_detalles}")
    
    logger.info("=== FIN DEL DIAGNÓSTICO ===")

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
    """Ejecuta la sincronización de forma programada."""
    # Código de sincronización existente...
    pass  # Mantenemos la función pero no la modificamos

def run_ml_forecast(train_new_model=True, save_model=True, generate_plots=True, train_product_models=True, top_products=10):
    """
    Ejecuta el módulo de forecasting con TensorFlow.
    
    Args:
        train_new_model: Si es True, entrena un nuevo modelo general
        save_model: Si es True, guarda los modelos entrenados
        generate_plots: Si es True, genera visualizaciones
        train_product_models: Si es True, entrena modelos por producto
        top_products: Número de productos top a predecir
    """
    logger.info("Iniciando módulo de forecasting...")
    
    # Ejecutar diagnóstico para entender qué datos están disponibles
    diagnosticar_datos_disponibles()
    
    # Obtenemos el cliente de MongoDB
    mongo = get_mongo_manager()
    
    # Creamos la instancia de nuestro modelo
    forecaster = TFForecaster(db_client=mongo)
    
    # Verificamos si hay un modelo general guardado
    model_path = "models/forecast"
    model_exists = os.path.exists(os.path.join(model_path, "tf_forecaster.h5"))
    
    # Verificamos si hay modelos de productos guardados
    product_model_path = "models/forecast/products"
    product_models_exist = os.path.exists(product_model_path) and len(os.listdir(product_model_path)) > 0
    
    try:
        # 1. MODELO GENERAL
        if model_exists and not train_new_model:
            logger.info("Cargando modelo general existente...")
            forecaster.load_model(path=model_path)
            logger.info("Modelo general cargado correctamente.")
        else:
            if train_new_model:
                logger.info("Iniciando entrenamiento de nuevo modelo general...")
                
                # Configurar memoria para TensorFlow (opcional)
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        logger.info(f"GPU disponible: {gpus}")
                        # Limitar memoria GPU si es necesario
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logger.warning(f"Error configurando GPU: {e}")
                else:
                    logger.info("Entrenando en CPU")
                
                # Entrenamiento del modelo general
                start_time = time.time()
                history = forecaster.train()
                end_time = time.time()
                training_time = round(end_time - start_time, 2)
                logger.info(f"Entrenamiento del modelo general completado en {training_time} segundos")
                
                # Guardar modelo general si es necesario
                if save_model:
                    logger.info("Guardando modelo general entrenado...")
                    os.makedirs(model_path, exist_ok=True)
                    forecaster.save_model(path=model_path)
                    logger.info(f"Modelo general guardado en {model_path}")
            else:
                logger.error("No se encontró un modelo general existente y no se solicitó entrenamiento.")
                return False
        
        # 2. MODELOS POR PRODUCTO
        if train_product_models:
            logger.info(f"Iniciando entrenamiento de modelos para los top {top_products} productos...")
            start_time = time.time()
            
            # Entrenar modelos por producto
            training_results = forecaster.train_product_models(top_n=top_products)
            
            end_time = time.time()
            training_time = round(end_time - start_time, 2)
            
            # Contar modelos exitosos
            success_count = sum(1 for r in training_results.values() if r.get('status') == 'success')
            logger.info(f"Entrenamiento de {success_count}/{len(training_results)} modelos de productos completado en {training_time} segundos")
            
            # Guardar modelos de productos si es necesario
            if save_model and success_count > 0:
                logger.info("Guardando modelos de productos entrenados...")
                os.makedirs(product_model_path, exist_ok=True)
                forecaster.save_product_models(path=product_model_path)
                logger.info(f"Modelos de productos guardados en {product_model_path}")
        elif product_models_exist:
            logger.info("Cargando modelos de productos existentes...")
            forecaster.load_product_models(path=product_model_path)
            logger.info("Modelos de productos cargados correctamente.")
        
        # 3. GENERAR PREDICCIONES GENERALES (DIARIAS, SEMANALES Y MENSUALES)
        logger.info("Generando predicciones generales...")
        
        # Predicciones diarias
        daily_predictions = forecaster.predict_next_days()
        logger.info(f"Predicciones diarias generadas para los próximos {len(daily_predictions)} días")
        
        # NUEVO: Predicciones semanales
        logger.info("Generando predicciones semanales...")
        weekly_predictions = forecaster.predict_aggregated(period='week', horizon=8)
        logger.info(f"Predicciones semanales generadas para las próximas {len(weekly_predictions)} semanas")
        
        # NUEVO: Predicciones mensuales
        logger.info("Generando predicciones mensuales...")
        monthly_predictions = forecaster.predict_aggregated(period='month', horizon=3)
        logger.info(f"Predicciones mensuales generadas para los próximos {len(monthly_predictions)} meses")
        
        # Guardar predicciones generales en MongoDB
        logger.info("Guardando predicciones diarias en MongoDB...")
        forecaster.save_predictions_to_db(daily_predictions)
        
        # NUEVO: Guardar predicciones semanales y mensuales en MongoDB
        try:
            # Formato para guardar datos de semanas y meses
            weekly_docs = [{
                "tipo": "semanal",
                "periodo": w["periodo"],
                "fecha_inicio": w["fecha_inicio"],
                "fecha_fin": w["fecha_fin"],
                "prediccion": w["prediccion"],
                "confianza": w["confianza"],
                "timestamp": datetime.now()
            } for w in weekly_predictions]
            
            monthly_docs = [{
                "tipo": "mensual",
                "periodo": m["periodo"],
                "fecha_inicio": m["fecha_inicio"],
                "fecha_fin": m["fecha_fin"],
                "prediccion": m["prediccion"],
                "confianza": m["confianza"],
                "timestamp": datetime.now()
            } for m in monthly_predictions]
            
            # Eliminar predicciones anteriores
            mongo.db[MONGO_PARAMS["collection_forecasts"]].delete_many({"tipo": "semanal"})
            mongo.db[MONGO_PARAMS["collection_forecasts"]].delete_many({"tipo": "mensual"})
            
            # Insertar nuevas predicciones
            if weekly_docs:
                mongo.db[MONGO_PARAMS["collection_forecasts"]].insert_many(weekly_docs)
                logger.info(f"Guardadas {len(weekly_docs)} predicciones semanales en MongoDB")
            
            if monthly_docs:
                mongo.db[MONGO_PARAMS["collection_forecasts"]].insert_many(monthly_docs)
                logger.info(f"Guardadas {len(monthly_docs)} predicciones mensuales en MongoDB")
        except Exception as e:
            logger.error(f"Error guardando predicciones agregadas: {str(e)}")
        
        # 4. GENERAR PREDICCIONES POR PRODUCTO
        if hasattr(forecaster, 'product_models') and forecaster.product_models:
            logger.info("Generando predicciones por producto...")
            product_predictions = forecaster.predict_product_demand()
            
            if not product_predictions.empty:
                logger.info(f"Predicciones generadas para {product_predictions['producto_id'].nunique()} productos")
                
                # Guardar predicciones por producto en MongoDB
                logger.info("Guardando predicciones por producto en MongoDB...")
                forecaster.save_product_predictions_to_db(product_predictions)
                
                # NUEVO: Generar predicciones por categoría
                logger.info("Generando predicciones por categoría...")
                category_predictions = forecaster.predict_category_demand()
                
                if category_predictions:
                    unique_categories = set(item['categoria_id'] for item in category_predictions)
                    logger.info(f"Predicciones generadas para {len(unique_categories)} categorías")
                    
                    # Guardar predicciones por categoría en MongoDB
                    try:
                        # Preparar documentos
                        category_docs = []
                        for pred in category_predictions:
                            doc = {
                                "fecha": datetime.strptime(pred["fecha"], '%Y-%m-%d') if isinstance(pred["fecha"], str) else pred["fecha"],
                                "dia": pred["dia"],
                                "categoria_id": pred["categoria_id"],
                                "nombre_categoria": pred["nombre_categoria"],
                                "prediccion": pred["prediccion"],
                                "confianza": pred["confianza"],
                                "productos": pred["productos"],
                                "timestamp": datetime.now()
                            }
                            category_docs.append(doc)
                        
                        # Eliminar predicciones anteriores
                        mongo.db[MONGO_PARAMS["collection_category_predictions"]].delete_many({})
                        
                        # Insertar nuevas predicciones
                        if category_docs:
                            mongo.db[MONGO_PARAMS["collection_category_predictions"]].insert_many(category_docs)
                            logger.info(f"Guardadas {len(category_docs)} predicciones por categoría en MongoDB")
                    except Exception as e:
                        logger.error(f"Error guardando predicciones por categoría: {str(e)}")
                else:
                    logger.warning("No se pudieron generar predicciones por categoría")
            else:
                logger.warning("No se pudieron generar predicciones por producto")
        
        # 5. GENERAR VISUALIZACIONES
        if generate_plots:
            # Crear directorio para gráficos si no existe
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generar y guardar gráfico general
            logger.info("Generando visualización de predicciones generales...")
            fig = forecaster.plot_forecast(history_days=30)
            plot_path = os.path.join(plots_dir, f"forecast_{time.strftime('%Y%m%d_%H%M%S')}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Visualización general guardada en {plot_path}")
            
            # MODIFICADO: Generar gráficos para TODOS los productos disponibles
            if hasattr(forecaster, 'product_models') and forecaster.product_models:
                # Usar todos los productos con modelos entrenados
                all_product_ids = list(forecaster.product_models.keys())
                logger.info(f"Generando visualizaciones para {len(all_product_ids)} productos...")
                
                for product_id in all_product_ids:
                    try:
                        logger.info(f"Generando visualización para producto {product_id}...")
                        fig = forecaster.plot_product_forecast(product_id, history_days=30)
                        product_plot_path = os.path.join(plots_dir, f"product_{product_id}_{time.strftime('%Y%m%d_%H%M%S')}.png")
                        fig.savefig(product_plot_path)
                        plt.close(fig)
                        logger.info(f"Visualización de producto {product_id} guardada en {product_plot_path}")
                    except Exception as e:
                        logger.warning(f"Error al generar visualización para producto {product_id}: {str(e)}")
                
                # NUEVO: Generar visualizaciones por categoría
                logger.info("Generando visualizaciones por categoría...")
                
                # Obtener categorías únicas de los productos modelados
                unique_categories = set()
                for product_id in forecaster.product_models:
                    product_info = forecaster.data_processor.get_product_info(product_id)
                    if 'categoria_id' in product_info and product_info['categoria_id'] is not None:
                        unique_categories.add(product_info['categoria_id'])
                
                logger.info(f"Generando visualizaciones para {len(unique_categories)} categorías...")
                
                # Generar gráfico para cada categoría
                for category_id in unique_categories:
                    try:
                        logger.info(f"Generando visualización para categoría {category_id}...")
                        fig = forecaster.plot_category_forecast(category_id, history_days=30)
                        category_plot_path = os.path.join(plots_dir, f"category_{category_id}_{time.strftime('%Y%m%d_%H%M%S')}.png")
                        fig.savefig(category_plot_path)
                        plt.close(fig)
                        logger.info(f"Visualización de categoría {category_id} guardada en {category_plot_path}")
                    except Exception as e:
                        logger.warning(f"Error al generar visualización para categoría {category_id}: {str(e)}")
        
        logger.info("Proceso de forecasting completado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error en el proceso de forecasting: {str(e)}", exc_info=True)
        return False

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
    
    try:
        # INSERTA TU CÓDIGO ML AQUÍ - REEMPLAZO CON NUESTRO CÓDIGO DE FORECASTING
        logger.info("Sincronización desactivada. Usando datos existentes en MongoDB para ML.")
        
        # Ejecutar módulo de forecasting
        ml_success = run_ml_forecast(
            train_new_model=True,        # True para entrenar nuevo modelo general
            save_model=True,             # True para guardar los modelos después de entrenar
            generate_plots=True,         # True para generar visualizaciones
            train_product_models=True,   # True para entrenar modelos por producto
            top_products=10              # Número de productos top a predecir
        )
        
        if ml_success:
            logger.info("Módulo ML ejecutado correctamente. Resultados disponibles en MongoDB.")
        else:
            logger.warning("El módulo ML presentó errores. Revisar logs para más detalles.")
        
        # Para entorno de desarrollo, podemos mantener el programa corriendo para probar más
        if config.dev_mode:
            logger.info("Modo desarrollo: Programa en ejecución. Presiona Ctrl+C para terminar.")
            while True:
                time.sleep(60)
        
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por el usuario")
    except Exception as e:
        logger.critical(f"Error fatal en el servicio: {str(e)}", exc_info=True)
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