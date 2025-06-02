import logging
from db.mongo_client import get_mongo_manager

logger = logging.getLogger(__name__)

def init_segmentation():
    """Inicializa el módulo de segmentación creando índices y verificando colecciones"""
    mongo = get_mongo_manager()
    
    # Verificar si las colecciones de segmentación existen
    collections = mongo.db.list_collection_names()
    required_collections = ['cliente_segmentacion', 'cluster_perfiles']
    
    missing_collections = [c for c in required_collections if c not in collections]
    if missing_collections:
        logger.warning(f"Faltan las siguientes colecciones para segmentación: {missing_collections}")
        logger.warning("Las consultas de segmentación podrían no estar disponibles hasta ejecutar el modelo")
    
    # Crear índices necesarios para segmentación
    try:
        # Índice para cliente_segmentacion
        mongo.db.cliente_segmentacion.create_index([("cliente_id", 1), ("tenant_id", 1)], unique=True)
        mongo.db.cliente_segmentacion.create_index([("cluster", 1), ("tenant_id", 1)])
        
        # Índice para cluster_perfiles
        mongo.db.cluster_perfiles.create_index([("cluster_id", 1), ("tenant_id", 1)], unique=True)
        
        logger.info("Índices de segmentación creados correctamente")
    except Exception as e:
        logger.error(f"Error creando índices de segmentación: {str(e)}")