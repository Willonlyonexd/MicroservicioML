import pandas as pd
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import logging

class RecommendationDataManager:
    def __init__(self, db_connection=None):
        """Inicializa el gestor de datos para recomendaciones"""
        self.db_connection = db_connection
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 3600  # 1 hora de caché
        self.logger = logging.getLogger('recommendation_data_manager')

    def get_client_purchase_history(self, client_id, tenant_id):
        """Obtiene el historial de compras de un cliente específico"""
        cache_key = f"history_{tenant_id}_{client_id}"
        if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            # Consulta corregida con manejo especial para valores booleanos en estado
            query = text("""
                SELECT p.producto_id, p.nombre, p.descripcion, p.precio, p.categoria_id,
                       cat.nombre as categoria_nombre,
                       COUNT(*) as frecuencia, MAX(pe.fecha_hora) as ultima_compra
                FROM pedido_detalle pd
                JOIN producto p ON pd.producto_id = p.producto_id
                JOIN pedido pe ON pd.pedido_id = pe.pedido_id
                JOIN cuenta_mesa cm ON pe.cuenta_mesa_id = cm.cuenta_mesa_id
                LEFT JOIN categoria cat ON p.categoria_id = cat.categoria_id
                WHERE cm.cliente_id = :client_id
                AND pe.tenant_id = :tenant_id
                AND (pe.estado = 'True' OR pe.estado = 't' OR pe.estado = 'true' OR pe.estado = '1' OR pe.estado IS NULL)
                GROUP BY p.producto_id, p.nombre, p.descripcion, p.precio, p.categoria_id, cat.nombre
                ORDER BY frecuencia DESC
            """)
            
            result = pd.read_sql(query, self.db_connection, params={
                'client_id': client_id,
                'tenant_id': tenant_id
            })
            
            # Verificar si se encontraron resultados
            if result.empty:
                self.logger.warning(f"No se encontraron compras para el cliente {client_id}")
            else:
                self.logger.info(f"Se encontraron {len(result)} productos comprados por el cliente {client_id}")
            
            # Convertir fechas a formato datetime si están como cadenas
            if 'ultima_compra' in result.columns and result['ultima_compra'].dtype == 'object':
                result['ultima_compra'] = pd.to_datetime(result['ultima_compra'], errors='coerce')
            
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_ttl
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error al obtener historial de compras: {str(e)}")
            return pd.DataFrame()
    
    def get_product_popularity(self, tenant_id, time_context=None, days=30):
        """Obtiene la popularidad de productos para un tenant"""
        cache_key = f"popularity_{tenant_id}_{days}_{str(time_context)}"
        if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            time_filter = ""
            query_params = {'tenant_id': tenant_id}
            
            if time_context:
                if 'dayOfWeek' in time_context:
                    time_filter += " AND EXTRACT(DOW FROM pe.fecha_hora::timestamp) = :day_of_week"
                    query_params['day_of_week'] = time_context['dayOfWeek']
                    
                if 'timeOfDay' in time_context:
                    time_ranges = {
                        "MORNING": "(EXTRACT(HOUR FROM pe.fecha_hora::timestamp) >= 6 AND EXTRACT(HOUR FROM pe.fecha_hora::timestamp) < 12)",
                        "LUNCH": "(EXTRACT(HOUR FROM pe.fecha_hora::timestamp) >= 12 AND EXTRACT(HOUR FROM pe.fecha_hora::timestamp) < 15)",
                        "AFTERNOON": "(EXTRACT(HOUR FROM pe.fecha_hora::timestamp) >= 15 AND EXTRACT(HOUR FROM pe.fecha_hora::timestamp) < 18)",
                        "DINNER": "(EXTRACT(HOUR FROM pe.fecha_hora::timestamp) >= 18 AND EXTRACT(HOUR FROM pe.fecha_hora::timestamp) < 23)"
                    }
                    if time_context['timeOfDay'] in time_ranges:
                        time_filter += f" AND {time_ranges[time_context['timeOfDay']]}"
            
            # Consulta mejorada para manejar todos los casos
            query = text(f"""
                SELECT p.producto_id, p.nombre, p.descripcion, p.precio, p.categoria_id,
                       cat.nombre as categoria_nombre,
                       COUNT(*) as pedidos_totales,
                       COUNT(DISTINCT pe.pedido_id) as pedidos_unicos,
                       CAST(AVG(pd.cantidad) AS FLOAT) as cantidad_promedio
                FROM pedido_detalle pd
                JOIN producto p ON pd.producto_id = p.producto_id
                JOIN pedido pe ON pd.pedido_id = pe.pedido_id
                LEFT JOIN categoria cat ON p.categoria_id = cat.categoria_id
                WHERE pe.tenant_id = :tenant_id
                AND (pe.estado = 'True' OR pe.estado = 't' OR pe.estado = 'true' OR pe.estado = '1' OR pe.estado IS NULL)
                {time_filter}
                GROUP BY p.producto_id, p.nombre, p.descripcion, p.precio, p.categoria_id, cat.nombre
                ORDER BY pedidos_totales DESC
                LIMIT 20
            """)
            
            self.logger.info(f"Ejecutando consulta de popularidad para tenant {tenant_id}")
            result = pd.read_sql(query, self.db_connection, params=query_params)
            
            # Verificar resultados
            if result.empty:
                self.logger.warning(f"No se encontraron productos populares para tenant {tenant_id}")
                # Fallback directo a consulta sin filtros de tiempo
                if time_context:
                    self.logger.info("Intentando consulta sin filtros de tiempo")
                    fallback_query = text(f"""
                        SELECT p.producto_id, p.nombre, p.descripcion, p.precio, p.categoria_id,
                            cat.nombre as categoria_nombre,
                            COUNT(*) as pedidos_totales,
                            COUNT(DISTINCT pe.pedido_id) as pedidos_unicos,
                            CAST(AVG(pd.cantidad) AS FLOAT) as cantidad_promedio
                        FROM pedido_detalle pd
                        JOIN producto p ON pd.producto_id = p.producto_id
                        JOIN pedido pe ON pd.pedido_id = pe.pedido_id
                        LEFT JOIN categoria cat ON p.categoria_id = cat.categoria_id
                        WHERE pe.tenant_id = :tenant_id
                        AND (pe.estado = 'True' OR pe.estado = 't' OR pe.estado = 'true' OR pe.estado = '1' OR pe.estado IS NULL)
                        GROUP BY p.producto_id, p.nombre, p.descripcion, p.precio, p.categoria_id, cat.nombre
                        ORDER BY pedidos_totales DESC
                        LIMIT 20
                    """)
                    result = pd.read_sql(fallback_query, self.db_connection, params={'tenant_id': tenant_id})
                    if not result.empty:
                        self.logger.info(f"Consulta sin filtros de tiempo encontró {len(result)} productos")
            else:
                self.logger.info(f"Se encontraron {len(result)} productos populares para tenant {tenant_id}")
            
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_ttl
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener popularidad de productos: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def get_product_details(self, product_ids, tenant_id):
        """Obtiene detalles de productos específicos"""
        if not product_ids:
            return pd.DataFrame()
            
        try:
            # Convertir IDs a lista si es necesario
            if not isinstance(product_ids, (list, tuple)):
                product_ids = [product_ids]
            
            # IMPORTANTE: Convertir de numpy.int64 a int de Python
            product_ids = [int(pid) for pid in product_ids]
            
            # Usar parámetros parametrizados correctamente con SQLAlchemy
            placeholders = ', '.join([f":id{i}" for i in range(len(product_ids))])
            params = {'tenant_id': tenant_id}
            for i, pid in enumerate(product_ids):
                params[f'id{i}'] = pid
                
            query = text(f"""
                SELECT p.producto_id, p.nombre, p.descripcion, p.precio, 
                       p.categoria_id, c.nombre as categoria_nombre
                FROM producto p
                LEFT JOIN categoria c ON p.categoria_id = c.categoria_id
                WHERE p.producto_id IN ({placeholders})
                AND p.tenant_id = :tenant_id
            """)
            
            result = pd.read_sql(query, self.db_connection, params=params)
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener detalles de productos: {str(e)}")
            return pd.DataFrame()
    
    def get_products_bought_together(self, tenant_id):
        """Obtiene productos que se compran juntos frecuentemente"""
        cache_key = f"together_{tenant_id}"
        if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            # Consulta modificada para manejar valores boolean en estado y sin restricción de frecuencia
            query = text("""
                SELECT pd1.producto_id as producto1_id, pd2.producto_id as producto2_id,
                       COUNT(*) as frecuencia_conjunta
                FROM pedido_detalle pd1
                JOIN pedido_detalle pd2 ON pd1.pedido_id = pd2.pedido_id AND pd1.producto_id < pd2.producto_id
                JOIN pedido pe ON pd1.pedido_id = pe.pedido_id
                WHERE pe.tenant_id = :tenant_id
                AND (pe.estado = 'True' OR pe.estado = 't' OR pe.estado = 'true' OR pe.estado = '1' OR pe.estado IS NULL)
                GROUP BY producto1_id, producto2_id
                ORDER BY frecuencia_conjunta DESC
                LIMIT 50
            """)
            
            self.logger.info(f"Consultando productos comprados juntos para tenant {tenant_id}")
            result = pd.read_sql(query, self.db_connection, params={'tenant_id': tenant_id})
            
            if result.empty:
                self.logger.warning(f"No se encontraron productos comprados juntos para tenant {tenant_id}")
            else:
                self.logger.info(f"Se encontraron {len(result)} pares de productos comprados juntos")
            
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_ttl
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener productos comprados juntos: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def get_time_based_recommendations(self, tenant_id, time_context):
        """Obtiene recomendaciones basadas en el momento del día o día de la semana"""
        try:
            return self.get_product_popularity(tenant_id, time_context)
        except Exception as e:
            self.logger.error(f"Error al obtener recomendaciones basadas en tiempo: {str(e)}")
            return pd.DataFrame()

    def clear_cache(self):
        """Limpia toda la caché de datos"""
        self.cache = {}
        self.cache_expiry = {}