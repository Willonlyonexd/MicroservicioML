    import pandas as pd
    import os
    import logging
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # Cache para los datos
    _segmentation_data = {}
    _cluster_info = {}

    def load_segmentation_data(tenant_id=1, mongo_client=None):
        """Carga los datos de segmentación desde MongoDB"""
        global _segmentation_data
        
        if tenant_id not in _segmentation_data or _segmentation_data[tenant_id] is None:
            try:
                if mongo_client:
                    # Cargar desde MongoDB usando el tenant_id
                    result = list(mongo_client.db.cliente_segmentacion.find({"tenant_id": tenant_id}))
                    if result:
                        df = pd.DataFrame(result)
                        _segmentation_data[tenant_id] = df
                        logger.info(f"Datos de segmentación para tenant {tenant_id} cargados de MongoDB: {len(df)} clientes")
                    else:
                        logger.warning(f"No se encontraron datos de segmentación en MongoDB para tenant {tenant_id}")
                        _segmentation_data[tenant_id] = pd.DataFrame()
                else:
                    logger.error("No se proporcionó cliente MongoDB")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error al cargar datos de segmentación: {str(e)}")
                return pd.DataFrame()
        
        return _segmentation_data.get(tenant_id, pd.DataFrame())

    def load_cluster_info(tenant_id=1, mongo_client=None):
        """Carga la información de los clusters desde MongoDB"""
        global _cluster_info
        
        if tenant_id not in _cluster_info or _cluster_info[tenant_id] is None:
            try:
                if mongo_client:
                    # Cargar desde MongoDB
                    result = list(mongo_client.db.cluster_perfiles.find({"tenant_id": tenant_id}))
                    if result:
                        _cluster_info[tenant_id] = result
                        logger.info(f"Información de clusters para tenant {tenant_id} cargada: {len(result)} clusters")
                    else:
                        logger.warning(f"No se encontró información de clusters para tenant {tenant_id}")
                        _cluster_info[tenant_id] = []
                else:
                    logger.error("No se proporcionó cliente MongoDB")
                    return []
            except Exception as e:
                logger.error(f"Error al cargar información de clusters: {str(e)}")
                return []
        
        return _cluster_info.get(tenant_id, [])

    # --- Resolvers para Ariadne ---

    def resolve_clientes_segmentados(obj, info, clusterId=None):
        """Resolver para obtener clientes segmentados"""
        try:
            from db.mongo_client import get_mongo_manager
            mongo = get_mongo_manager()
            
            tenant_id = info.context.get('tenant_id', 1)
            df = load_segmentation_data(tenant_id, mongo)
            
            if df.empty:
                logger.warning(f"No hay datos de segmentación para tenant {tenant_id}")
                return []
            
            # Filtrar por cluster si se especifica
            if clusterId is not None:
                df = df[df['cluster'] == clusterId]
            
            # Convertir a formato para GraphQL
            result = []
            for _, row in df.iterrows():
                result.append({
                    'clienteId': str(int(row['cliente_id'])),
                    'cluster': int(row['cluster']),
                    'clusterNombre': row.get('cluster_nombre', "DESCONOCIDO"),
                    'numVisitas': int(row['num_visitas']),
                    'comensalesPromedio': float(row['comensales_promedio']),
                    'gastoTotal': float(row['gasto_total']),
                    'tenantId': tenant_id
                })
            
            return result
        except Exception as e:
            logger.error(f"Error en resolve_clientes_segmentados: {str(e)}")
            return []

    def resolve_cluster_info(obj, info, clusterId):
        """Resolver para obtener información de un cluster específico"""
        try:
            from db.mongo_client import get_mongo_manager
            mongo = get_mongo_manager()
            
            tenant_id = info.context.get('tenant_id', 1)
            clusters = load_cluster_info(tenant_id, mongo)
            
            for cluster in clusters:
                if cluster['cluster_id'] == clusterId:
                    return {
                        'clusterId': cluster['cluster_id'],
                        'nombre': cluster['nombre'],
                        'numClientes': cluster['num_clientes'],
                        'gastoPromedio': cluster['gasto_promedio'],
                        'visitasPromedio': cluster['visitas_promedio'],
                        'comensalesPromedio': cluster['comensales_promedio'],
                        'descripcion': cluster['descripcion'],
                        'tenantId': tenant_id
                    }
            
            logger.warning(f"No se encontró información para el cluster {clusterId} del tenant {tenant_id}")
            return None
        except Exception as e:
            logger.error(f"Error en resolve_cluster_info: {str(e)}")
            return None

    def resolve_todos_clusters(obj, info):
        """Resolver para obtener información de todos los clusters"""
        try:
            from db.mongo_client import get_mongo_manager
            mongo = get_mongo_manager()
            
            tenant_id = info.context.get('tenant_id', 1)
            clusters = load_cluster_info(tenant_id, mongo)
            
            result = []
            for cluster in clusters:
                result.append({
                    'clusterId': cluster['cluster_id'],
                    'nombre': cluster['nombre'],
                    'numClientes': cluster['num_clientes'],
                    'gastoPromedio': cluster['gasto_promedio'],
                    'visitasPromedio': cluster['visitas_promedio'],
                    'comensalesPromedio': cluster['comensales_promedio'],
                    'descripcion': cluster['descripcion'],
                    'tenantId': tenant_id
                })
            
            return result
        except Exception as e:
            logger.error(f"Error en resolve_todos_clusters: {str(e)}")
            return []