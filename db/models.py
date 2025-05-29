from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from utils import now

class BaseDocument:
    """Clase base para documentos de MongoDB."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Crea una instancia desde un diccionario."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la instancia a un diccionario."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class SyncStatus(BaseDocument):
    """Estado de sincronización de una tabla."""
    
    def __init__(self, tenant_id: int, table_name: str, 
                last_sync: datetime = None, records_synced: int = 0,
                status: str = "pending", error: str = None):
        self.tenant_id = tenant_id
        self.table_name = table_name
        self.last_sync = last_sync or now()
        self.records_synced = records_synced
        self.status = status  # pending, running, completed, error
        self.error = error
        self.created_at = now()
        self.updated_at = now()

class RawPedido(BaseDocument):
    """Modelo para pedidos raw importados desde el ERP."""
    
    def __init__(self, tenant_id: int, pedido_id: int, cuenta_mesa_id: int,
                fecha_hora: datetime, total: float, estado: str,
                created_at: datetime = None, updated_at: datetime = None, **kwargs):
        self.tenant_id = tenant_id
        self.pedido_id = pedido_id  # ID original en ERP
        self.cuenta_mesa_id = cuenta_mesa_id
        self.fecha_hora = fecha_hora
        self.total = total
        self.estado = estado
        self.created_at = created_at or now()
        self.updated_at = updated_at or now()
        self.last_sync = now()
        
        # Campos adicionales
        for key, value in kwargs.items():
            setattr(self, key, value)

class RawPedidoDetalle(BaseDocument):
    """Modelo para detalles de pedido raw importados desde el ERP."""
    
    def __init__(self, tenant_id: int, pedido_detalle_id: int, pedido_id: int,
                producto_id: int, cantidad: int, subtotal: float, estado: str,
                created_at: datetime = None, updated_at: datetime = None, **kwargs):
        self.tenant_id = tenant_id
        self.pedido_detalle_id = pedido_detalle_id  # ID original en ERP
        self.pedido_id = pedido_id
        self.producto_id = producto_id
        self.cantidad = cantidad
        self.subtotal = subtotal
        self.estado = estado
        self.created_at = created_at or now()
        self.updated_at = updated_at or now()
        self.last_sync = now()
        
        # Campos adicionales
        for key, value in kwargs.items():
            setattr(self, key, value)

class RawProducto(BaseDocument):
    """Modelo para productos raw importados desde el ERP."""
    
    def __init__(self, tenant_id: int, producto_id: int, nombre: str,
                precio: float, categoria_id: int, disponible: bool,
                created_at: datetime = None, updated_at: datetime = None, **kwargs):
        self.tenant_id = tenant_id
        self.producto_id = producto_id  # ID original en ERP
        self.nombre = nombre
        self.precio = precio
        self.categoria_id = categoria_id
        self.disponible = disponible
        self.created_at = created_at or now()
        self.updated_at = updated_at or now()
        self.last_sync = now()
        
        # Campos adicionales
        for key, value in kwargs.items():
            setattr(self, key, value)

class VentasDiarias(BaseDocument):
    """Modelo para datos analíticos de ventas diarias."""
    
    def __init__(self, tenant_id: int, fecha: datetime, producto_id: int,
                nombre_producto: str, categoria_id: int, nombre_categoria: str,
                unidades_vendidas: int, monto_total: float, 
                dia_semana: int, mes: int, dia_mes: int, 
                computed_at: datetime = None):
        self.tenant_id = tenant_id
        self.fecha = fecha
        self.producto_id = producto_id
        self.nombre_producto = nombre_producto
        self.categoria_id = categoria_id
        self.nombre_categoria = nombre_categoria
        self.unidades_vendidas = unidades_vendidas
        self.monto_total = monto_total
        self.dia_semana = dia_semana  # 0=domingo, 6=sábado
        self.mes = mes  # 1-12
        self.dia_mes = dia_mes  # 1-31
        self.computed_at = computed_at or now()

# Crear índices recomendados para MongoDB
# Al final de db/models.py

def create_indexes(mongo_manager=None):
    """Crea índices para las colecciones de MongoDB."""
    from db.mongo_client import get_mongo_manager
    from utils import logger
    
    try:
        # Usar el gestor pasado o obtener el singleton
        mongo = mongo_manager if mongo_manager is not None else get_mongo_manager()
        
        # Índices para tablas raw - el formato correcto es list de tuplas
        mongo.create_index("raw_pedidos", [("tenant_id", 1), ("pedido_id", 1)], unique=True)
        mongo.create_index("raw_pedidos", [("tenant_id", 1), ("fecha_hora", -1)])
        
        mongo.create_index("raw_pedido_detalles", [("tenant_id", 1), ("pedido_detalle_id", 1)], unique=True)
        mongo.create_index("raw_pedido_detalles", [("tenant_id", 1), ("pedido_id", 1)])
        mongo.create_index("raw_pedido_detalles", [("tenant_id", 1), ("producto_id", 1)])
        
        mongo.create_index("raw_productos", [("tenant_id", 1), ("producto_id", 1)], unique=True)
        mongo.create_index("raw_productos", [("tenant_id", 1), ("categoria_id", 1)])
        
        # Índices para tablas analíticas
        mongo.create_index("ventas_diarias", [("tenant_id", 1), ("fecha", 1), ("producto_id", 1)], unique=True)
        mongo.create_index("ventas_diarias", [("tenant_id", 1), ("producto_id", 1)])
        mongo.create_index("ventas_diarias", [("tenant_id", 1), ("categoria_id", 1)])
        
        # Índices para estado de sincronización
        mongo.create_index("sync_status", [("tenant_id", 1), ("table_name", 1)], unique=True)
        
        logger.info("Índices creados para todas las colecciones")
    except Exception as e:
        logger.error(f"Error al crear índices: {str(e)}")
        raise