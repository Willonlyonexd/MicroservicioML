from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from utils import logger, timing_decorator, tenant_aware, now
from db.mongo_client import get_mongo_manager
from config import get_config

class DataLoader:
    """Cargador de datos transformados a MongoDB."""
    
    def __init__(self):
        """Inicializa el cargador de datos."""
        self.config = get_config()
        self.mongo = get_mongo_manager()
    
    def _get_collection_name(self, table_name: str) -> str:
        """Determina el nombre de colección en MongoDB según la tabla.
        
        Args:
            table_name: Nombre de la tabla
            
        Returns:
            Nombre de colección
        """
        # Mapeo de tablas a colecciones
        collection_map = {
            "pedido": "raw_pedidos",
            "pedido_detalle": "raw_pedido_detalles",
            "producto": "raw_productos",
            "cliente": "raw_clientes",
            "categoria": "raw_categorias",
            "usuario": "raw_usuarios",
            "mesa": "raw_mesas",
            "cuenta_mesa": "raw_cuenta_mesas",
            "reserva": "raw_reservas",
            "venta": "raw_ventas"
        }
        
        return collection_map.get(table_name, f"raw_{table_name}s")
    
    def _get_key_field(self, table_name: str) -> str:
        """Determina el campo clave para upsert según la tabla.
        
        Args:
            table_name: Nombre de la tabla
            
        Returns:
            Nombre del campo clave
        """
        # Mapeo de tablas a campos clave
        key_fields = {
            "pedido": "pedido_id",
            "pedido_detalle": "pedido_detalle_id",
            "producto": "producto_id",
            "cliente": "cliente_id",
            "categoria": "categoria_id",
            "usuario": "usuario_id",
            "mesa": "mesa_id",
            "cuenta_mesa": "cuenta_mesa_id",
            "reserva": "reserva_id", 
            "venta": "venta_id"
        }
        
        # Usar campo específico o tenant_id + id como fallback
        return key_fields.get(table_name, "id")
    
    @timing_decorator
    @tenant_aware
    def load_records(self, tenant_id: int, table_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Carga registros transformados en MongoDB.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            records: Registros transformados
            
        Returns:
            Resultado de la operación
        """
        if not records:
            return {"status": "success", "inserted": 0, "updated": 0}
            
        try:
            # Determinar colección
            collection_name = self._get_collection_name(table_name)
            
            # Determinar campo clave
            key_field = self._get_key_field(table_name)
            
            # Verificar que todos los registros tengan tenant_id
            for record in records:
                if "tenant_id" not in record:
                    record["tenant_id"] = tenant_id
                    
                # Asegurar que timestamp esté presente
                if "etl_timestamp" not in record:
                    record["etl_timestamp"] = now()
            
            # Insertar o actualizar registros
            # CORREGIDO: Pasar argumentos en el orden correcto sin nombres
            result = self.mongo.upsert_many(
                collection_name,
                records,
                key_field
            )
            
            logger.info(f"Cargados {len(records)} registros en {collection_name}: {result.get('inserted', 0)} insertados, {result.get('updated', 0)} actualizados")
            
            return result
            
        except Exception as e:
            logger.error(f"Error cargando registros en {table_name}: {str(e)}")
            raise
    
    @timing_decorator
    @tenant_aware
    def update_sync_status(self, tenant_id: int, table_name: str, 
                         records_synced: int = 0, status: str = "success", 
                         error: Optional[str] = None) -> Dict[str, Any]:
        """Actualiza el estado de sincronización de una tabla.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            records_synced: Cantidad de registros sincronizados
            status: Estado de la sincronización
            error: Mensaje de error, si ocurrió
            
        Returns:
            Documento de estado de sincronización
        """
        sync_status = {
            "tenant_id": tenant_id,
            "table_name": table_name,
            "last_sync": now(),
            "records_synced": records_synced,
            "status": status,
            "error": error,
            "updated_at": now()
        }
        
        # Actualizar o insertar estado de sincronización
        self.mongo.update_one(
            "sync_status",
            {"tenant_id": tenant_id, "table_name": table_name},
            {"$set": sync_status},
            upsert=True
        )
        
        logger.info(f"Estado de sincronización actualizado para {table_name}, tenant {tenant_id}")
        return sync_status
    
    @timing_decorator
    @tenant_aware
    def get_sync_status(self, tenant_id: int, table_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de sincronización de una tabla.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            
        Returns:
            Documento de estado o None
        """
        try:
            # Buscar en MongoDB
            result = self.mongo.find_one(
                "sync_status",
                {"tenant_id": tenant_id, "table_name": table_name}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de sincronización: {str(e)}")
            return None

# Singleton
data_loader = DataLoader()

# Función para obtener el cargador de datos
def get_data_loader():
    return data_loader