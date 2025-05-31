from datetime import datetime, timedelta
import os
import glob
import logging
from db.mongo_client import get_mongo_manager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obtener cliente de MongoDB
mongo = get_mongo_manager()

# Mapeo correcto de colecciones
COLLECTION_DAILY = "predicciones_ventas"
COLLECTION_WEEKLY_MONTHLY = "ml_predicciones"
COLLECTION_PRODUCTS = "predicciones_productos"
COLLECTION_CATEGORIES = "ml_predicciones_categoria"

def resolve_daily_forecasts(obj, info, days=7):
    """Resolver para obtener predicciones diarias"""
    try:
        logger.info(f"Buscando {days} predicciones diarias en {COLLECTION_DAILY}")
        
        # Verificar si la colección existe
        if COLLECTION_DAILY not in mongo.db.list_collection_names():
            logger.error(f"Colección {COLLECTION_DAILY} no existe")
            return []
        
        # Buscar predicciones diarias con tipo "general"
        predictions = list(mongo.db[COLLECTION_DAILY].find(
            {"tipo": "general"},
            {"_id": 0}
        ).sort("fecha", 1).limit(days))
        
        logger.info(f"Encontradas {len(predictions)} predicciones diarias")
        
        # Mapear campos para coincidir con el schema GraphQL
        result = []
        for pred in predictions:
            if isinstance(pred.get('fecha'), datetime):
                fecha = pred['fecha'].isoformat()
            else:
                fecha = pred.get('fecha')
                
            result.append({
                "fecha": fecha,
                "prediccion": pred.get('prediccion'),
                "confianza": pred.get('confianza'),
                "timestamp": pred.get('timestamp').isoformat() if isinstance(pred.get('timestamp'), datetime) else pred.get('timestamp'),
                "generado_en": pred.get('timestamp').isoformat() if isinstance(pred.get('timestamp'), datetime) else pred.get('timestamp'),
                "generado_por": "ML System v1.0 - muimui69"
            })
        
        return result
    except Exception as e:
        logger.error(f"Error obteniendo predicciones diarias: {str(e)}")
        return []

def resolve_weekly_forecasts(obj, info, weeks=1):
    """Resolver para obtener predicciones semanales"""
    try:
        logger.info(f"Buscando {weeks} predicciones semanales en {COLLECTION_WEEKLY_MONTHLY}")
        
        # Buscar predicciones semanales
        predictions = list(mongo.db[COLLECTION_WEEKLY_MONTHLY].find(
            {"tipo": "semanal"},
            {"_id": 0}
        ).sort("periodo", 1).limit(weeks))
        
        logger.info(f"Encontradas {len(predictions)} predicciones semanales")
        
        # Convertir fechas a string si son datetime
        for pred in predictions:
            if isinstance(pred.get('fecha_inicio'), datetime):
                pred['fecha_inicio'] = pred['fecha_inicio'].isoformat()
            if isinstance(pred.get('fecha_fin'), datetime):
                pred['fecha_fin'] = pred['fecha_fin'].isoformat()
            if isinstance(pred.get('timestamp'), datetime):
                pred['timestamp'] = pred['timestamp'].isoformat()
        
        return predictions
    except Exception as e:
        logger.error(f"Error obteniendo predicciones semanales: {str(e)}")
        return []

def resolve_monthly_forecasts(obj, info, months=1):
    """Resolver para obtener predicciones mensuales"""
    try:
        logger.info(f"Buscando {months} predicciones mensuales en {COLLECTION_WEEKLY_MONTHLY}")
        
        # Buscar predicciones mensuales
        predictions = list(mongo.db[COLLECTION_WEEKLY_MONTHLY].find(
            {"tipo": "mensual"},
            {"_id": 0}
        ).sort("periodo", 1).limit(months))
        
        logger.info(f"Encontradas {len(predictions)} predicciones mensuales")
        
        # Convertir fechas a string si son datetime
        for pred in predictions:
            if isinstance(pred.get('fecha_inicio'), datetime):
                pred['fecha_inicio'] = pred['fecha_inicio'].isoformat()
            if isinstance(pred.get('fecha_fin'), datetime):
                pred['fecha_fin'] = pred['fecha_fin'].isoformat()
            if isinstance(pred.get('timestamp'), datetime):
                pred['timestamp'] = pred['timestamp'].isoformat()
        
        return predictions
    except Exception as e:
        logger.error(f"Error obteniendo predicciones mensuales: {str(e)}")
        return []

def resolve_product_forecast(obj, info, productId, days=7):
    """Resolver para obtener predicciones de un producto específico"""
    try:
        logger.info(f"Buscando predicciones para producto {productId} en {COLLECTION_PRODUCTS}")
        
        # Buscar predicciones para el producto específico
        predictions = list(mongo.db[COLLECTION_PRODUCTS].find(
            {"producto_id": productId},
            {"_id": 0}
        ).sort("fecha", 1).limit(days))
        
        logger.info(f"Encontradas {len(predictions)} predicciones para el producto {productId}")
        
        # Si no hay predicciones, buscar info del producto para devolver estructura básica
        if not predictions:
            product_info = mongo.db["raw_productos"].find_one({"producto_id": productId})
            if not product_info:
                logger.warning(f"No se encontró información para el producto {productId}")
                return None
                
            return {
                "producto_id": productId,
                "nombre": product_info.get("nombre", f"Producto {productId}"),
                "categoria_id": product_info.get("categoria_id", 0),
                "categoria": "Desconocida",
                "predicciones": []
            }
        
        # Procesar resultados
        processed_predictions = []
        for pred in predictions:
            if isinstance(pred.get('fecha'), datetime):
                fecha = pred['fecha'].isoformat()
            else:
                fecha = pred.get('fecha')
                
            processed_predictions.append({
                "fecha": fecha,
                "prediccion": pred.get('prediccion'),
                "confianza": pred.get('confianza')
            })
        
        # Buscar info adicional del producto si está disponible
        categoria = "Platos Principales"  # Default
        if "categoria_id" in predictions[0]:
            categoria_id = predictions[0]["categoria_id"]
            categoria_info = mongo.db["raw_categorias"].find_one({"categoria_id": categoria_id})
            if categoria_info and "nombre" in categoria_info:
                categoria = categoria_info["nombre"]
        
        return {
            "producto_id": productId,
            "nombre": predictions[0].get("nombre_producto", f"Producto {productId}"),
            "categoria_id": predictions[0].get("categoria_id", 0),
            "categoria": categoria,
            "predicciones": processed_predictions
        }
    except Exception as e:
        logger.error(f"Error obteniendo predicciones del producto {productId}: {str(e)}")
        return None

def resolve_category_forecast(obj, info, categoryId, days=7):
    """Resolver para obtener predicciones de una categoría específica"""
    try:
        logger.info(f"Buscando predicciones para categoría {categoryId} en {COLLECTION_CATEGORIES}")
        
        # Ver qué categorías existen
        categorias = mongo.db[COLLECTION_CATEGORIES].distinct("categoria_id")
        logger.info(f"Categorías disponibles: {categorias}")
        
        # Buscar predicciones para la categoría específica
        predictions = list(mongo.db[COLLECTION_CATEGORIES].find(
            {"categoria_id": categoryId},
            {"_id": 0}
        ).sort("fecha", 1).limit(days))
        
        logger.info(f"Encontradas {len(predictions)} predicciones para la categoría {categoryId}")
        
        if not predictions:
            logger.warning(f"No se encontraron predicciones para la categoría {categoryId}")
            return None
            
        # Procesar resultados
        processed_predictions = []
        for pred in predictions:
            if isinstance(pred.get('fecha'), datetime):
                fecha = pred['fecha'].isoformat()
            else:
                fecha = pred.get('fecha')
                
            processed_predictions.append({
                "fecha": fecha,
                "prediccion": pred.get('prediccion'),
                "confianza": pred.get('confianza'),
                "productos": pred.get('productos', [])
            })
        
        return {
            "categoria_id": categoryId,
            "nombre_categoria": predictions[0].get("nombre_categoria", f"Categoría {categoryId}"),
            "predicciones": processed_predictions
        }
    except Exception as e:
        logger.error(f"Error obteniendo predicciones de la categoría {categoryId}: {str(e)}")
        return None

def resolve_model_status(obj, info):
    """Resolver para obtener el estado de los modelos"""
    try:
        logger.info("Obteniendo estado de los modelos")
        
        # Verificar modelo general
        model_path = "models/forecast/tf_forecaster.h5"
        general_model_exists = os.path.exists(model_path)
        general_model_time = datetime.fromtimestamp(os.path.getmtime(model_path)) if general_model_exists else None
        
        # Estado del modelo general
        general_status = {
            "entrenado": general_model_exists,
            "ultima_actualizacion": general_model_time.isoformat() if general_model_time else None,
            "exactitud": 0.85,  # Valores de ejemplo
            "error_mae": 0.15
        }
        
        # Verificar modelos de productos
        product_models_path = "models/forecast/products"
        product_models = []
        
        if os.path.exists(product_models_path):
            # Buscar archivos de modelos
            model_files = glob.glob(os.path.join(product_models_path, "product_*_model.h5"))
            logger.info(f"Encontrados {len(model_files)} modelos de productos")
            
            for model_file in model_files:
                # Extraer ID de producto del nombre de archivo
                file_name = os.path.basename(model_file)
                product_id = int(file_name.split("_")[1])
                
                # Obtener timestamp del archivo
                model_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                
                # Buscar información del producto
                product_info = mongo.db["raw_productos"].find_one({"producto_id": product_id})
                product_name = product_info.get("nombre", f"Producto {product_id}") if product_info else f"Producto {product_id}"
                
                # Añadir información del modelo
                product_models.append({
                    "producto_id": product_id,
                    "nombre": product_name,
                    "entrenado": True,
                    "ultima_actualizacion": model_time.isoformat(),
                    "exactitud": 0.80  # Valor de ejemplo
                })
        else:
            logger.warning(f"Directorio de modelos de productos no encontrado: {product_models_path}")
        
        return {
            "general": general_status,
            "productos": product_models
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado de los modelos: {str(e)}")
        return {"general": {"entrenado": False, "exactitud": 0}, "productos": []}

def resolve_available_visualizations(obj, info):
    """Resolver para obtener las visualizaciones disponibles"""
    try:
        logger.info("Obteniendo visualizaciones disponibles")
        
        plots_dir = "plots"
        result = {"general": None, "productos": [], "categorias": []}
        
        if not os.path.exists(plots_dir):
            logger.warning(f"Directorio de visualizaciones no encontrado: {plots_dir}")
            return result
        
        # Buscar visualización general
        general_plots = sorted(glob.glob(os.path.join(plots_dir, "forecast_*.png")))
        if general_plots:
            latest_plot = general_plots[-1]
            plot_time = datetime.fromtimestamp(os.path.getmtime(latest_plot))
            result["general"] = {
                "url": f"/static/plots/{os.path.basename(latest_plot)}",
                "fecha_generacion": plot_time.isoformat()
            }
            logger.info(f"Visualización general encontrada: {os.path.basename(latest_plot)}")
        
        # Buscar visualizaciones por producto
        product_plots = glob.glob(os.path.join(plots_dir, "product_*_*.png"))
        logger.info(f"Encontradas {len(product_plots)} visualizaciones de productos")
        
        for plot_file in product_plots:
            file_name = os.path.basename(plot_file)
            parts = file_name.split("_")
            if len(parts) >= 3:
                try:
                    product_id = int(parts[1])
                    plot_time = datetime.fromtimestamp(os.path.getmtime(plot_file))
                    
                    # Buscar información del producto
                    product_info = mongo.db["raw_productos"].find_one({"producto_id": product_id})
                    product_name = product_info.get("nombre", f"Producto {product_id}") if product_info else f"Producto {product_id}"
                    
                    result["productos"].append({
                        "producto_id": product_id,
                        "nombre": product_name,
                        "url": f"/static/plots/{file_name}",
                        "fecha_generacion": plot_time.isoformat()
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error procesando visualización de producto {file_name}: {str(e)}")
                    continue
        
        # Buscar visualizaciones por categoría
        category_plots = glob.glob(os.path.join(plots_dir, "category_*_*.png"))
        logger.info(f"Encontradas {len(category_plots)} visualizaciones de categorías")
        
        for plot_file in category_plots:
            file_name = os.path.basename(plot_file)
            parts = file_name.split("_")
            if len(parts) >= 3:
                try:
                    category_id = int(parts[1])
                    plot_time = datetime.fromtimestamp(os.path.getmtime(plot_file))
                    
                    # Buscar nombre de categoría
                    categoria = mongo.db["raw_categorias"].find_one({"categoria_id": category_id})
                    category_name = categoria.get("nombre", f"Categoría {category_id}") if categoria else f"Categoría {category_id}"
                    
                    result["categorias"].append({
                        "categoria_id": category_id,
                        "nombre": category_name,
                        "url": f"/static/plots/{file_name}",
                        "fecha_generacion": plot_time.isoformat()
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error procesando visualización de categoría {file_name}: {str(e)}")
                    continue
        
        return result
    except Exception as e:
        logger.error(f"Error obteniendo visualizaciones disponibles: {str(e)}")
        return {"general": None, "productos": [], "categorias": []}

# Diccionario de resolvers para Ariadne
resolvers = {
    "Query": {
        "dailyForecasts": resolve_daily_forecasts,
        "weeklyForecasts": resolve_weekly_forecasts,
        "monthlyForecasts": resolve_monthly_forecasts,
        "productForecast": resolve_product_forecast,
        "categoryForecast": resolve_category_forecast,
        "modelStatus": resolve_model_status,
        "availableVisualizations": resolve_available_visualizations
    }
}