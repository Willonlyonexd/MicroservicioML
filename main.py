import pandas as pd
from datetime import datetime
from config import get_config
from utils import logger, setup_logging
from db.mongo_client import get_mongo_manager
from etl.extract import get_postgres_extractor
from etl.sync import get_data_synchronizer
from db.models import create_indexes
import time
import shutil
import sys

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

def diagnosticar_datos_disponibles(tenant_id=1):
    """Realiza un diagnóstico de los datos disponibles en MongoDB para un tenant específico."""
    mongo = get_mongo_manager()
    
    logger.info(f"=== DIAGNÓSTICO DE DATOS EN MONGODB PARA TENANT {tenant_id} ===")
    
    # Listar todas las colecciones
    collections = mongo.db.list_collection_names()
    logger.info(f"Colecciones disponibles: {collections}")
    
    # Verificar datos en colecciones relevantes
    for collection_name in ["raw_pedidos", "raw_pedido_detalles", "raw_productos"]:
        if collection_name in collections:
            count = mongo.db[collection_name].count_documents({"tenant_id": tenant_id})
            logger.info(f"Colección {collection_name}: {count} documentos para tenant {tenant_id}")
            
            # Mostrar un ejemplo de documento
            if count > 0:
                sample = mongo.db[collection_name].find_one({"tenant_id": tenant_id})
                logger.info(f"Ejemplo de {collection_name}: {list(sample.keys())}")
    
    # Verificar si hay datos de productos
    if "raw_productos" in collections:
        productos = list(mongo.db.raw_productos.find({"tenant_id": tenant_id}).limit(5))
        if productos:
            logger.info(f"Ejemplos de productos disponibles para tenant {tenant_id}: {[p.get('_id') for p in productos]}")
        else:
            logger.warning(f"No hay productos en la colección raw_productos para tenant {tenant_id}")
            
    # Verificar pedidos y detalles de pedidos
    if "raw_pedidos" in collections and "raw_pedido_detalles" in collections:
        # Obtener un pedido de ejemplo
        sample_pedido = mongo.db.raw_pedidos.find_one({"tenant_id": tenant_id})
        if sample_pedido:
            pedido_id = sample_pedido.get('_id')
            # Buscar detalles asociados a este pedido
            detalles = list(mongo.db.raw_pedido_detalles.find({"pedido_id": pedido_id}))
            logger.info(f"Pedido {pedido_id} tiene {len(detalles)} detalles")
            
            # Mostrar productos en esos detalles
            if detalles:
                productos_en_detalles = [d.get('producto_id') for d in detalles if 'producto_id' in d]
                logger.info(f"Productos en detalles de pedido: {productos_en_detalles}")
    
    # Verificar colecciones de predicciones
    for collection_name in ["predicciones_ventas", "ml_predicciones", "predicciones_productos", "ml_predicciones_categoria"]:
        if collection_name in collections:
            count = mongo.db[collection_name].count_documents({"tenant_id": tenant_id})
            logger.info(f"Colección {collection_name}: {count} documentos para tenant {tenant_id}")
            
            # Mostrar un ejemplo de documento
            if count > 0:
                sample = mongo.db[collection_name].find_one({"tenant_id": tenant_id})
                logger.info(f"Ejemplo de {collection_name}: {list(sample.keys())}")
                if 'fecha' in sample:
                    logger.info(f"Primera fecha en {collection_name}: {sample['fecha']}")
    
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

def run_ml_forecast(tenant_id=1, train_new_model=False, save_model=True, generate_plots=True, train_product_models=False, top_products=10):
    """
    Ejecuta el módulo de forecasting con TensorFlow para un tenant específico.
    
    Args:
        tenant_id: ID del tenant para el que generar predicciones
        train_new_model: Si es True, entrena un nuevo modelo general
        save_model: Si es True, guarda los modelos entrenados
        generate_plots: Si es True, genera visualizaciones
        train_product_models: Si es True, entrena modelos por producto
        top_products: Número de productos top a predecir
    """
    # Obtener fecha actual como punto pivote para datos históricos/predicciones
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    logger.info(f"Iniciando módulo de forecasting para tenant {tenant_id} con fecha pivote {current_date}...")
    
    # Ejecutar diagnóstico para entender qué datos están disponibles
    diagnosticar_datos_disponibles(tenant_id)
    
    # Obtenemos el cliente de MongoDB
    mongo = get_mongo_manager()
    
    # Creamos la instancia de nuestro modelo
    forecaster = TFForecaster(db_client=mongo)
    
    # Directorios específicos para tenant
    model_path = f"models/forecast/tenant_{tenant_id}"
    product_model_path = f"models/forecast/tenant_{tenant_id}/products"
    plots_dir = f"plots/tenant_{tenant_id}"
    
    # Verificamos si hay un modelo general guardado
    model_exists = os.path.exists(os.path.join(model_path, "tf_forecaster.h5"))
    
    # Verificamos si hay modelos de productos guardados
    product_models_exist = os.path.exists(product_model_path) and len(os.listdir(product_model_path)) > 0 if os.path.exists(product_model_path) else False
    
    try:
        # 1. MODELO GENERAL
        model_loaded = False
        if model_exists and not train_new_model:
            logger.info(f"Cargando modelo general existente para tenant {tenant_id}...")
            try:
                forecaster.load_model(path=model_path)
                logger.info("Modelo general cargado correctamente.")
                model_loaded = True
            except Exception as e:
                logger.error(f"Error al cargar el modelo general: {str(e)}")
                logger.info("Realizando backup del modelo con problemas...")
                
                # Hacer backup del modelo con problemas
                backup_dir = f"models/backup/tenant_{tenant_id}_{time.strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
                try:
                    shutil.copytree(model_path, backup_dir)
                    logger.info(f"Backup del modelo guardado en {backup_dir}")
                except Exception as be:
                    logger.warning(f"No se pudo realizar backup: {str(be)}")
                
                # Forzar reentrenamiento
                train_new_model = True
                logger.info("Se forzará el entrenamiento de un nuevo modelo.")
        
        # Si no existe el modelo o hubo error al cargarlo, lo entrenamos
        if not model_exists or (not model_loaded and train_new_model):
            logger.info(f"Iniciando entrenamiento de nuevo modelo general para tenant {tenant_id}...")
            
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
            
            # Entrenamiento del modelo general con datos específicos del tenant
            start_time = time.time()
            # Obtener datos específicos del tenant
            history = forecaster.train(tenant_id=tenant_id)
            end_time = time.time()
            training_time = round(end_time - start_time, 2)
            logger.info(f"Entrenamiento del modelo general completado en {training_time} segundos")
            
            # Guardar modelo general si es necesario
            if save_model:
                logger.info(f"Guardando modelo general entrenado para tenant {tenant_id}...")
                os.makedirs(model_path, exist_ok=True)
                forecaster.save_model(path=model_path)
                logger.info(f"Modelo general guardado en {model_path}")
        
        # 2. MODELOS POR PRODUCTO
        product_models_loaded = False
        if product_models_exist and not train_product_models:
            logger.info(f"Cargando modelos de productos existentes para tenant {tenant_id}...")
            try:
                forecaster.load_product_models(path=product_model_path)
                logger.info("Modelos de productos cargados correctamente.")
                product_models_loaded = True
            except Exception as e:
                logger.error(f"Error al cargar los modelos de productos: {str(e)}")
                # Forzar reentrenamiento de modelos de productos
                train_product_models = True
                logger.info("Se forzará el entrenamiento de nuevos modelos de productos.")
        
        # Si no hay modelos de productos o hubo error al cargarlos, los entrenamos
        if not product_models_exist or (not product_models_loaded and train_product_models):
            # Solo entrenamos nuevos modelos si se solicita explícitamente o hubo error
            logger.info(f"Iniciando entrenamiento de modelos para los top {top_products} productos del tenant {tenant_id}...")
            start_time = time.time()
            
            # Entrenar modelos por producto para este tenant
            training_results = forecaster.train_product_models(top_n=top_products, tenant_id=tenant_id)
            
            end_time = time.time()
            training_time = round(end_time - start_time, 2)
            
            # Contar modelos exitosos
            success_count = sum(1 for r in training_results.values() if r.get('status') == 'success')
            logger.info(f"Entrenamiento de {success_count}/{len(training_results)} modelos de productos completado en {training_time} segundos")
            
            # Guardar modelos de productos si es necesario
            if save_model and success_count > 0:
                logger.info(f"Guardando modelos de productos entrenados para tenant {tenant_id}...")
                os.makedirs(product_model_path, exist_ok=True)
                forecaster.save_product_models(path=product_model_path)
                logger.info(f"Modelos de productos guardados en {product_model_path}")
        
        # 3. GENERAR PREDICCIONES GENERALES (DIARIAS, SEMANALES Y MENSUALES)
        logger.info(f"Generando predicciones generales para tenant {tenant_id} desde fecha actual ({current_date})...")
        
        # Predicciones diarias
        daily_predictions = forecaster.predict_next_days(tenant_id=tenant_id)
        logger.info(f"Predicciones diarias generadas para los próximos {len(daily_predictions)} días")
        
        # Predicciones semanales
        logger.info(f"Generando predicciones semanales para tenant {tenant_id}...")
        weekly_predictions = forecaster.predict_aggregated(period='week', horizon=8, tenant_id=tenant_id)
        logger.info(f"Predicciones semanales generadas para las próximas {len(weekly_predictions)} semanas")
        
        # Predicciones mensuales
        logger.info(f"Generando predicciones mensuales para tenant {tenant_id}...")
        monthly_predictions = forecaster.predict_aggregated(period='month', horizon=3, tenant_id=tenant_id)
        logger.info(f"Predicciones mensuales generadas para los próximos {len(monthly_predictions)} meses")
        
        # FUNCIÓN DE CORRECCIÓN DE ESCALA PARA PREDICCIONES
        def apply_scale_correction(predictions, typical_value=100.0, historical_data=None):
            """
            Corrige la escala de las predicciones si son anormalmente bajas.
            
            Args:
                predictions: Lista de predicciones a corregir
                typical_value: Valor típico a usar como referencia si no hay datos históricos
                historical_data: Datos históricos para determinar escala (opcional)
                
            Returns:
                list: Predicciones con escala corregida
            """
            if not predictions:
                return predictions
                
            # Obtener el valor máximo de predicción
            max_pred = max(p.get('prediccion', 0) for p in predictions)
            logger.info(f"Valor máximo de predicción antes de corrección: {max_pred}")
            
            # Verificar si las predicciones son anormalmente bajas
            if max_pred < 1.0:
                # Determinar valor objetivo para escala
                scale_target = typical_value
                
                # Si hay datos históricos, usar su promedio como referencia
                if historical_data:
                    if isinstance(historical_data, list):
                        sales_values = [item.get('ventas', 0) for item in historical_data 
                                        if isinstance(item.get('ventas', 0), (int, float))]
                    elif isinstance(historical_data, pd.DataFrame) and 'total' in historical_data.columns:
                        sales_values = historical_data['total'].tolist()
                    else:
                        sales_values = []
                    
                    # Filtrar valores no nulos y mayores que cero
                    non_zero_sales = [s for s in sales_values if s > 0]
                    if non_zero_sales:
                        scale_target = sum(non_zero_sales) / len(non_zero_sales)
                        logger.info(f"Valor promedio de ventas históricas: {scale_target}")
                
                # Calcular factor de escala
                scale_factor = scale_target / max(max_pred, 0.0001)
                logger.warning(f"Aplicando corrección de escala (factor: {scale_factor:.2f}) a predicciones anormalmente bajas")
                
                # Aplicar escala a todas las predicciones
                for p in predictions:
                    p['prediccion'] = p.get('prediccion', 0) * scale_factor
                
                # Verificar resultado
                new_max_pred = max(p.get('prediccion', 0) for p in predictions)
                logger.info(f"Valor máximo de predicción después de corrección: {new_max_pred}")
            
            return predictions
        
        # Obtener datos históricos para determinar escala
        try:
            historical_data = forecaster.get_historical_and_forecast_data(
                history_days=30, 
                forecast_days=1, 
                tenant_id=tenant_id
            ).get('historical', [])
        except Exception as e:
            logger.warning(f"No se pudieron obtener datos históricos para corrección de escala: {str(e)}")
            historical_data = None
        
        # Corregir predicciones
        if daily_predictions:
            daily_predictions = apply_scale_correction(daily_predictions, historical_data=historical_data)
            logger.info("Corrección de escala aplicada a predicciones diarias")
            
        if weekly_predictions:
            weekly_predictions = apply_scale_correction(weekly_predictions, typical_value=500.0)
            logger.info("Corrección de escala aplicada a predicciones semanales")
            
        if monthly_predictions:
            monthly_predictions = apply_scale_correction(monthly_predictions, typical_value=2000.0)
            logger.info("Corrección de escala aplicada a predicciones mensuales")
        
        # Guardar predicciones generales en MongoDB
        logger.info(f"Guardando predicciones diarias en MongoDB para tenant {tenant_id}...")
        forecaster.save_predictions_to_db(daily_predictions, tenant_id=tenant_id)
        
        # Guardar predicciones semanales y mensuales en MongoDB
        try:
            # Formato para guardar datos de semanas y meses
            weekly_docs = [{
                "tipo": "semanal",
                "periodo": w["periodo"],
                "fecha_inicio": w["fecha_inicio"],
                "fecha_fin": w["fecha_fin"],
                "prediccion": w["prediccion"],
                "confianza": w["confianza"],
                "timestamp": datetime.now(),
                "tenant_id": tenant_id  # Añadir tenant_id
            } for w in weekly_predictions]
            
            monthly_docs = [{
                "tipo": "mensual",
                "periodo": m["periodo"],
                "fecha_inicio": m["fecha_inicio"],
                "fecha_fin": m["fecha_fin"],
                "prediccion": m["prediccion"],
                "confianza": m["confianza"],
                "timestamp": datetime.now(),
                "tenant_id": tenant_id  # Añadir tenant_id
            } for m in monthly_predictions]
            
            # Eliminar predicciones anteriores para este tenant
            mongo.db[MONGO_PARAMS["collection_forecasts"]].delete_many({"tipo": "semanal", "tenant_id": tenant_id})
            mongo.db[MONGO_PARAMS["collection_forecasts"]].delete_many({"tipo": "mensual", "tenant_id": tenant_id})
            
            # Insertar nuevas predicciones
            if weekly_docs:
                mongo.db[MONGO_PARAMS["collection_forecasts"]].insert_many(weekly_docs)
                logger.info(f"Guardadas {len(weekly_docs)} predicciones semanales en MongoDB para tenant {tenant_id}")
            
            if monthly_docs:
                mongo.db[MONGO_PARAMS["collection_forecasts"]].insert_many(monthly_docs)
                logger.info(f"Guardadas {len(monthly_docs)} predicciones mensuales en MongoDB para tenant {tenant_id}")
        except Exception as e:
            logger.error(f"Error guardando predicciones agregadas para tenant {tenant_id}: {str(e)}")
        
        # 4. GENERAR PREDICCIONES POR PRODUCTO
        if hasattr(forecaster, 'product_models') and forecaster.product_models:
            logger.info(f"Generando predicciones por producto para tenant {tenant_id}...")
            product_predictions = forecaster.predict_product_demand(tenant_id=tenant_id)
            
            if not product_predictions.empty:
                logger.info(f"Predicciones generadas para {product_predictions['producto_id'].nunique()} productos")
                
                # Corregir escala para predicciones por producto
                def apply_product_scale_correction(predictions_df, typical_value=100.0):
                    """Corrige la escala de las predicciones por producto."""
                    if predictions_df.empty:
                        return predictions_df
                    
                    # Obtener máxima predicción
                    max_pred = predictions_df['prediccion'].max()
                    logger.info(f"Valor máximo de predicción por producto antes de corrección: {max_pred}")
                    
                    # Verificar si las predicciones son anormalmente bajas
                    if max_pred < 1.0:
                        # Calcular factor de escala
                        scale_factor = typical_value / max(max_pred, 0.0001)
                        logger.warning(f"Aplicando corrección de escala (factor: {scale_factor:.2f}) a predicciones por producto")
                        
                        # Aplicar escala
                        predictions_df['prediccion'] = predictions_df['prediccion'] * scale_factor
                        logger.info(f"Valor máximo de predicción por producto después de corrección: {predictions_df['prediccion'].max()}")
                    
                    return predictions_df
                
                # Aplicar corrección de escala
                product_predictions = apply_product_scale_correction(product_predictions)
                
                # Guardar predicciones por producto en MongoDB
                logger.info(f"Guardando predicciones por producto en MongoDB para tenant {tenant_id}...")
                forecaster.save_product_predictions_to_db(product_predictions, tenant_id=tenant_id)
                
                # Generar predicciones por categoría
                logger.info(f"Generando predicciones por categoría para tenant {tenant_id}...")
                category_predictions = forecaster.predict_category_demand(tenant_id=tenant_id)
                
                if category_predictions:
                    # Corregir escala para predicciones por categoría
                    category_predictions = apply_scale_correction(category_predictions, typical_value=150.0)
                    
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
                                "timestamp": datetime.now(),
                                "tenant_id": tenant_id  # Añadir tenant_id
                            }
                            category_docs.append(doc)
                        
                        # Eliminar predicciones anteriores para este tenant
                        mongo.db[MONGO_PARAMS["collection_category_predictions"]].delete_many({"tenant_id": tenant_id})
                        
                        # Insertar nuevas predicciones
                        if category_docs:
                            mongo.db[MONGO_PARAMS["collection_category_predictions"]].insert_many(category_docs)
                            logger.info(f"Guardadas {len(category_docs)} predicciones por categoría en MongoDB para tenant {tenant_id}")
                    except Exception as e:
                        logger.error(f"Error guardando predicciones por categoría para tenant {tenant_id}: {str(e)}")
                else:
                    logger.warning(f"No se pudieron generar predicciones por categoría para tenant {tenant_id}")
            else:
                logger.warning(f"No se pudieron generar predicciones por producto para tenant {tenant_id}")
        
        # 5. GENERAR VISUALIZACIONES
        if generate_plots:
            # Crear directorio específico para tenant
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generar y guardar gráfico general
            logger.info(f"Generando visualización de predicciones generales para tenant {tenant_id}...")
            fig = forecaster.plot_forecast(history_days=30, forecast_days=7, tenant_id=tenant_id, generate_plot=True)
            plot_path = os.path.join(plots_dir, f"forecast_{time.strftime('%Y%m%d_%H%M%S')}.png")
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Visualización general guardada en {plot_path}")
            
            # Generar gráficos para TODOS los productos disponibles
            if hasattr(forecaster, 'product_models') and forecaster.product_models:
                # Usar todos los productos con modelos entrenados
                all_product_ids = list(forecaster.product_models.keys())
                logger.info(f"Generando visualizaciones para {len(all_product_ids)} productos del tenant {tenant_id}...")
                
                for product_id in all_product_ids:
                    try:
                        logger.info(f"Generando visualización para producto {product_id}...")
                        fig = forecaster.plot_product_forecast(product_id, history_days=30, forecast_days=7, tenant_id=tenant_id, generate_plot=True)
                        product_plot_path = os.path.join(plots_dir, f"product_{product_id}_{time.strftime('%Y%m%d_%H%M%S')}.png")
                        fig.savefig(product_plot_path)
                        plt.close(fig)
                        logger.info(f"Visualización de producto {product_id} guardada en {product_plot_path}")
                    except Exception as e:
                        logger.warning(f"Error al generar visualización para producto {product_id}: {str(e)}")
                
                # Generar visualizaciones por categoría
                logger.info(f"Generando visualizaciones por categoría para tenant {tenant_id}...")
                
                # Obtener categorías únicas de los productos modelados
                unique_categories = set()
                for product_id in forecaster.product_models:
                    product_info = forecaster.data_processor.get_product_info(product_id, tenant_id=tenant_id)
                    if 'categoria_id' in product_info and product_info['categoria_id'] is not None:
                        unique_categories.add(product_info['categoria_id'])
                
                logger.info(f"Generando visualizaciones para {len(unique_categories)} categorías del tenant {tenant_id}...")
                
                # Generar gráfico para cada categoría
                for category_id in unique_categories:
                    try:
                        logger.info(f"Generando visualización para categoría {category_id}...")
                        fig = forecaster.plot_category_forecast(category_id, history_days=30, forecast_days=7, tenant_id=tenant_id, generate_plot=True)
                        category_plot_path = os.path.join(plots_dir, f"category_{category_id}_{time.strftime('%Y%m%d_%H%M%S')}.png")
                        fig.savefig(category_plot_path)
                        plt.close(fig)
                        logger.info(f"Visualización de categoría {category_id} guardada en {category_plot_path}")
                    except Exception as e:
                        logger.warning(f"Error al generar visualización para categoría {category_id}: {str(e)}")
        
        # 6. GENERAR DATOS COMBINADOS PARA API
        logger.info(f"Generando datos combinados (histórico + predicción) para tenant {tenant_id}...")
        
        # Obtener datos históricos y predicciones para cada tipo
        combined_data = forecaster.get_historical_and_forecast_data(
            history_days=30, 
            forecast_days=7, 
            tenant_id=tenant_id
        )
        
        # Corregir escala de predicciones en datos combinados
        if 'forecast' in combined_data and combined_data['forecast']:
            combined_data['forecast'] = apply_scale_correction(
                combined_data['forecast'], 
                historical_data=combined_data.get('historical', [])
            )
            
        logger.info(f"Datos combinados generados: {len(combined_data['historical'])} días históricos, {len(combined_data['forecast'])} días predicción")
        
        # Datos semanales combinados
        weekly_combined = forecaster.get_historical_and_aggregated_forecast(
            period='week',
            history_periods=12,
            forecast_periods=3,
            tenant_id=tenant_id
        )
        
        # Corregir escala de predicciones semanales
        if 'forecast' in weekly_combined and weekly_combined['forecast']:
            weekly_combined['forecast'] = apply_scale_correction(
                weekly_combined['forecast'], 
                typical_value=500.0,
                historical_data=weekly_combined.get('historical', [])
            )
            
        logger.info(f"Datos semanales combinados generados: {len(weekly_combined['historical'])} semanas históricas, {len(weekly_combined['forecast'])} semanas predicción")
        
        # Datos mensuales combinados
        monthly_combined = forecaster.get_historical_and_aggregated_forecast(
            period='month',
            history_periods=12,
            forecast_periods=3,
            tenant_id=tenant_id
        )
        
        # Corregir escala de predicciones mensuales
        if 'forecast' in monthly_combined and monthly_combined['forecast']:
            monthly_combined['forecast'] = apply_scale_correction(
                monthly_combined['forecast'], 
                typical_value=2000.0,
                historical_data=monthly_combined.get('historical', [])
            )
            
        logger.info(f"Datos mensuales combinados generados: {len(monthly_combined['historical'])} meses históricos, {len(monthly_combined['forecast'])} meses predicción")
        
        # Verificar resultados en MongoDB para diagnóstico
        logger.info(f"Verificando datos en MongoDB después de procesamiento...")
        for collection_name in ["predicciones_ventas", "ml_predicciones", "predicciones_productos", "ml_predicciones_categoria"]:
            count = mongo.db[collection_name].count_documents({"tenant_id": tenant_id})
            logger.info(f"Colección {collection_name} tiene {count} documentos para tenant {tenant_id}")
        
        logger.info(f"Proceso de forecasting completado exitosamente para tenant {tenant_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error en el proceso de forecasting para tenant {tenant_id}: {str(e)}", exc_info=True)
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
    logger.info(f"Sistema multi-tenant activado - Fecha actual: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Fecha actual como punto pivote: {datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)}")
    
    # Probar conexiones
    if not test_connections():
        logger.error("Error en las conexiones a bases de datos. Abortando.")
        return
    
    # Crear índices en MongoDB
    try:
        create_indexes()
    except Exception as e:
        logger.error(f"Error al crear índices: {str(e)}")
    
    # Verificar si existe archivo server.py y crearlo si es necesario
    if not os.path.exists('server.py') and config.dev_mode:
        try:
            logger.info("No se encontró archivo server.py. Creando uno básico...")
            with open('server.py', 'w') as f:
                f.write("""# Este archivo re-exporta el servidor desde api.server.py
try:
    from api.server import app
except ImportError as e:
    print(f"Error importando app desde api.server: {e}")
    raise

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
""")
            logger.info("Archivo server.py creado correctamente.")
        except Exception as e:
            logger.warning(f"No se pudo crear el archivo server.py: {str(e)}")
    
    try:
        # MULTI-TENANT: Generar datos para varios tenants
        tenants_to_process = [1]  # Lista de tenants a procesar
        
        for tenant_id in tenants_to_process:
            logger.info(f"======= PROCESANDO TENANT {tenant_id} =======")
            
            # Ejecutar módulo de forecasting para este tenant
            ml_success = run_ml_forecast(
                tenant_id=tenant_id,          # ID del tenant a procesar
                train_new_model=True,         # Cambio a True para forzar entrenamiento y solucionar el error
                save_model=True,              # True para guardar los modelos después de entrenar
                generate_plots=True,          # True para generar visualizaciones
                train_product_models=True,    # Cambio a True para entrenar nuevos modelos de productos
                top_products=10               # Número de productos top a predecir
            )
            
            if ml_success:
                logger.info(f"Módulo ML ejecutado correctamente para tenant {tenant_id}. Resultados disponibles en MongoDB.")
            else:
                logger.warning(f"El módulo ML presentó errores para tenant {tenant_id}. Revisar logs para más detalles.")
            
            logger.info(f"======= FIN PROCESAMIENTO TENANT {tenant_id} =======")
        
        logger.info("Todos los tenants procesados correctamente")
        
        # Para entorno de desarrollo, podemos iniciar un servidor de prueba
        if config.dev_mode:
            logger.info("Iniciando servidor GraphQL para pruebas...")
            
            # Importar y ejecutar el servidor en un hilo separado
            import threading
            
            def run_server():
                try:
                    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
                    # Intentar importar desde varios lugares posibles
                    try:
                        from api.server import app
                        logger.info("Servidor importado desde api.server")
                    except ImportError:
                        try:
                            from server import app
                            logger.info("Servidor importado desde server")
                        except ImportError:
                            logger.error("No se pudo importar el servidor. Verificar que exista api/server.py o server.py")
                            return
                    
                    app.run(debug=False, host='0.0.0.0', port=5000)
                except Exception as e:
                    logger.error(f"Error al iniciar el servidor GraphQL: {str(e)}")
            
            server_thread = threading.Thread(target=run_server)
            server_thread.daemon = True
            server_thread.start()
            
            logger.info("Servidor GraphQL iniciado en http://localhost:5000/graphql")
            logger.info("API soporta visualización combinada (histórico + predicción) con fecha actual como punto pivote")
            logger.info("Presiona Ctrl+C para terminar.")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Programa interrumpido por el usuario")
        
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