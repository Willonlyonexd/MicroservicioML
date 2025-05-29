import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text
from config import get_config
from utils import logger, timing_decorator, tenant_aware
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from db.mongo_client import get_mongo_manager

# Base para modelos SQLAlchemy
Base = declarative_base()

class PostgresExtractor:
    """Extractor de datos desde PostgreSQL (sistema ERP)."""
    
    def __init__(self):
        """Inicializa el extractor de PostgreSQL."""
        self.config = get_config()
        self.engine = None
        self.Session = None
        self._connect()
        
        # Mapeo de nombres de columnas de ID primaria por tabla
        self.id_columns = {
            "pedido": "pedido_id",
            "pedido_detalle": "pedido_detalle_id",
            "producto": "producto_id",
            "cliente": "cliente_id",
            "mesa": "mesa_id",
            "categoria": "categoria_id",
            "usuario": "usuario_id",
            "cuenta_mesa": "cuenta_mesa_id",
            "reserva": "reserva_id",
            "venta": "venta_id",
            "comanda": "comanda_id",
            "rol": "rol_id",
            "permiso": "permiso_id",
            "unidad_medida": "unidad_medida_id",
            "insumo": "insumo_id",
            "almacen": "almacen_id"
        }
        
        # Mapeo de nombres de colecciones en MongoDB
        self.collection_names = {
            "pedido": "raw_pedidos",
            "pedido_detalle": "raw_pedido_detalles",
            "producto": "raw_productos",
            "cliente": "raw_clientes",
            "mesa": "raw_mesas",
            "categoria": "raw_categorias",
            "usuario": "raw_usuarios",
            "cuenta_mesa": "raw_cuenta_mesas",
            "reserva": "raw_reservas",
            "venta": "raw_ventas",
            "comanda": "raw_comandas",
            "rol": "raw_roles",
            "permiso": "raw_permisos",
            "unidad_medida": "raw_unidad_medidas",
            "insumo": "raw_insumos",
            "almacen": "raw_almacenes"
        }
    
    @timing_decorator
    def _connect(self) -> None:
        """Establece conexión con PostgreSQL."""
        try:
            # Crear engine de SQLAlchemy
            self.engine = sa.create_engine(
                self.config.database.postgres_uri,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                connect_args={"connect_timeout": 10}
            )
            
            # Probar conexión
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() != 1:
                    raise Exception("Test de conexión falló")
            
            # Crear fábrica de sesiones
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info(f"Conexión a PostgreSQL establecida: {self.config.database.postgres_uri}")
        except Exception as e:
            logger.error(f"No se pudo conectar a PostgreSQL: {str(e)}")
            raise
    
    def get_session(self):
        """Obtiene una sesión de SQLAlchemy."""
        if not self.Session:
            self._connect()
        return self.Session()
    
    @timing_decorator
    @tenant_aware
    def extract_table(self, tenant_id: int, table_name: str, 
                     last_sync: Optional[datetime] = None, 
                     batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extrae datos de una tabla usando sincronización inteligente basada en IDs.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            last_sync: Fecha de última sincronización (usado solo para logging)
            batch_size: No utilizado, se mantiene por compatibilidad
            
        Returns:
            Lista de registros extraídos como diccionarios
        """
        # Determinar el nombre de la columna ID para esta tabla
        id_column = self.id_columns.get(table_name)
        collection_name = self.collection_names.get(table_name, f"raw_{table_name}s")
        
        if not id_column:
            logger.warning(f"No se ha definido la columna ID para la tabla {table_name}, usando extracción completa")
            return self._extract_all(tenant_id, table_name)
        
        with self.get_session() as session:
            try:
                # Si es la primera sincronización, extraer todo
                if not last_sync:
                    logger.info(f"Sincronización completa inicial para {table_name}")
                    return self._extract_all(tenant_id, table_name)
                
                logger.info(f"Sincronización incremental para {table_name} (última: {last_sync})")
                
                # Paso 1: Obtener todos los IDs existentes en PostgreSQL
                id_query = f"SELECT {id_column} FROM {table_name} WHERE tenant_id = :tenant_id"
                id_result = session.execute(text(id_query), {"tenant_id": tenant_id})
                pg_ids = {row[0] for row in id_result}  # Conjunto de IDs en PostgreSQL
                
                if not pg_ids:
                    logger.info(f"No hay registros en PostgreSQL para {table_name}")
                    return []
                
                # Paso 2: Obtener IDs existentes en MongoDB
                mongo = get_mongo_manager()
                mongo_ids_cursor = mongo.db[collection_name].find(
                    {"tenant_id": tenant_id}, 
                    {id_column: 1, "_id": 0}
                )
                mongo_ids = {doc.get(id_column) for doc in mongo_ids_cursor if doc.get(id_column)}
                
                # Informar sobre el análisis de IDs
                logger.info(f"Análisis para {table_name}: {len(pg_ids)} registros en PostgreSQL, {len(mongo_ids)} en MongoDB")
                
                # CAMBIO CLAVE: Verificar si hay cambios entre MongoDB y PostgreSQL
                if len(pg_ids) == len(mongo_ids):
                    # Si hay el mismo número de registros, verificar si son los mismos IDs
                    if pg_ids == mongo_ids:
                        logger.info(f"No hay cambios detectados en {table_name} desde la última sincronización")
                        return []  # No hay cambios, retornar lista vacía
                
                # Calcular IDs que necesitan sincronización (nuevos o no existentes en MongoDB)
                new_ids = pg_ids - mongo_ids
                
                # Si hay nuevos IDs, extraerlos
                if new_ids:
                    logger.info(f"Detectados {len(new_ids)} registros nuevos en {table_name}")
                    
                    # Si hay muchos nuevos IDs, podría ser más eficiente extraer todo
                    if len(new_ids) > 1000:
                        logger.info(f"Muchos registros nuevos, extrayendo todos para {table_name}")
                        return self._extract_all(tenant_id, table_name)
                    
                    # Extraer solo los registros nuevos
                    id_list = list(new_ids)
                    query = f"SELECT * FROM {table_name} WHERE tenant_id = :tenant_id AND {id_column} IN :ids"
                    result = session.execute(
                        text(query), 
                        {"tenant_id": tenant_id, "ids": tuple(id_list)}
                    )
                    records = [dict(row._mapping) for row in result]
                    logger.info(f"Extraídos {len(records)} registros nuevos de {table_name}")
                    return records
                else:
                    # Si hay menos IDs en PostgreSQL que en MongoDB, algunos registros fueron eliminados
                    # Por ahora, simplemente extraemos todos para asegurar consistencia
                    # En el futuro se podría implementar una estrategia más sofisticada
                    if len(pg_ids) < len(mongo_ids):
                        logger.info(f"Detectados registros eliminados en {table_name}, actualizando")
                        return self._extract_all(tenant_id, table_name)
                    
                    # Si llegamos aquí, hay el mismo número de IDs pero no son exactamente los mismos
                    # Extraer todo para asegurar consistencia
                    logger.info(f"Detectados cambios en los IDs de {table_name}, actualizando")
                    return self._extract_all(tenant_id, table_name)
                
            except Exception as e:
                logger.error(f"Error en la estrategia incremental para {table_name}: {str(e)}")
                # Si hay error en la estrategia incremental, caer en modo de respaldo
                logger.info(f"Intentando extracción completa como respaldo para {table_name}")
                return self._extract_all(tenant_id, table_name)
    
    @timing_decorator
    def _extract_all(self, tenant_id: int, table_name: str) -> List[Dict[str, Any]]:
        """Extrae todos los registros de una tabla para un tenant específico.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            
        Returns:
            Lista de registros extraídos como diccionarios
        """
        with self.get_session() as session:
            try:
                query = f"SELECT * FROM {table_name} WHERE tenant_id = :tenant_id"
                result = session.execute(text(query), {"tenant_id": tenant_id})
                records = [dict(row._mapping) for row in result]
                logger.info(f"Extraídos {len(records)} registros de {table_name} para tenant {tenant_id}")
                return records
            except Exception as e:
                logger.error(f"Error al extraer todos los datos de {table_name}: {str(e)}")
                session.rollback()
                raise
    
    @timing_decorator
    @tenant_aware
    def extract_table_by_ids(self, tenant_id: int, table_name: str, 
                           ids: List[int], id_column: str = None) -> List[Dict[str, Any]]:
        """Extrae registros específicos por IDs.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            ids: Lista de IDs a extraer
            id_column: Nombre de la columna ID (opcional, se intentará determinar automáticamente)
            
        Returns:
            Lista de registros extraídos como diccionarios
        """
        if not ids:
            return []
        
        # Si no se especifica id_column, intentar determinarlo
        if id_column is None:
            id_column = self.id_columns.get(table_name)
            if not id_column:
                logger.warning(f"No se ha definido la columna ID para la tabla {table_name}")
                return []
            
        with self.get_session() as session:
            try:
                # Construir consulta SQL
                query = f"""
                SELECT * FROM {table_name} 
                WHERE tenant_id = :tenant_id 
                AND {id_column} IN :ids
                """
                
                # Ejecutar consulta
                result = session.execute(
                    text(query), 
                    {"tenant_id": tenant_id, "ids": tuple(ids)}
                )
                
                # Convertir a lista de diccionarios
                records = [dict(row._mapping) for row in result]
                
                logger.info(f"Extraídos {len(records)} registros específicos de {table_name}")
                return records
                
            except Exception as e:
                logger.error(f"Error al extraer datos específicos de {table_name}: {str(e)}")
                session.rollback()
                raise
    
    @timing_decorator
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ejecuta una consulta SQL personalizada.
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            Lista de registros como diccionarios
        """
        with self.get_session() as session:
            try:
                result = session.execute(text(query), params or {})
                records = [dict(row._mapping) for row in result]
                return records
            except Exception as e:
                logger.error(f"Error al ejecutar consulta personalizada: {str(e)}")
                session.rollback()
                raise
    
    @timing_decorator
    @tenant_aware
    def get_latest_sync_status(self, tenant_id: int, table_name: str) -> Optional[datetime]:
        """Obtiene la fecha de última sincronización de una tabla desde MongoDB.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            
        Returns:
            Fecha de última sincronización o None
        """
        try:
            # Obtener la fecha de última sincronización desde MongoDB
            mongo = get_mongo_manager()
            status = mongo.db.sync_status.find_one(
                {"tenant_id": tenant_id, "table_name": table_name}
            )
            
            if status and status.get("last_sync_time"):
                last_sync = status["last_sync_time"]
                logger.info(f"Última sincronización de {table_name} para tenant {tenant_id}: {last_sync}")
                return last_sync
            
            logger.info(f"No hay registro de sincronización previa para {table_name}, tenant {tenant_id}")
            return None
        except Exception as e:
            logger.error(f"Error al obtener estado de sincronización para {table_name}: {str(e)}")
            return None  # En caso de error, forzar sincronización completa
    
    def close(self) -> None:
        """Cierra la conexión con PostgreSQL."""
        if self.engine:
            self.engine.dispose()
            logger.info("Conexión a PostgreSQL cerrada")

# Singleton
postgres_extractor = PostgresExtractor()

# Función para obtener el extractor de PostgreSQL
def get_postgres_extractor() -> PostgresExtractor:
    return postgres_extractor