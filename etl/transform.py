from datetime import datetime
from typing import Dict, List, Any, Optional
from utils import logger, timing_decorator, tenant_aware, now
from db.models import RawPedido, RawPedidoDetalle, RawProducto, SyncStatus
from config import get_config
import datetime as dt  # Importación explícita para evitar confusiones

class DataTransformer:
    """Transformador de datos entre formato ERP (PostgreSQL) y data warehouse (MongoDB)."""
    
    def __init__(self):
        """Inicializa el transformador de datos."""
        self.config = get_config()
    
    @timing_decorator
    @tenant_aware
    def transform_pedido(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de pedido.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Convertir fechas si vienen como strings
            fecha_hora = record.get("fecha_hora")
            if isinstance(fecha_hora, str):
                fecha_hora = datetime.fromisoformat(fecha_hora.replace('Z', '+00:00'))
            
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "pedido_id": record.get("pedido_id"),
                "cuenta_mesa_id": record.get("cuenta_mesa_id"),
                "fecha_hora": fecha_hora,
                "fecha": fecha_hora.date().isoformat() if fecha_hora else None,  # Extraer solo la fecha para agregaciones
                "hora": fecha_hora.time().isoformat() if fecha_hora else None,   # Extraer solo la hora para análisis de patrones
                "dia_semana": fecha_hora.weekday() if fecha_hora else None,      # 0=Lunes, 6=Domingo
                "total": float(record.get("total", 0)),
                "nota": record.get("nota", ""),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de pedido")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_pedido_detalle(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de detalle de pedido.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "pedido_detalle_id": record.get("pedido_detalle_id"),
                "pedido_id": record.get("pedido_id"),
                "producto_id": record.get("producto_id"),
                "cantidad": int(record.get("cantidad", 0)),
                "subtotal": float(record.get("subtotal", 0)),
                "precio_unitario": float(record.get("subtotal", 0)) / int(record.get("cantidad", 1)) if int(record.get("cantidad", 0)) > 0 else 0,
                "estado": record.get("estado", ""),
                "nota": record.get("nota", ""),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de detalle de pedido")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_producto(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de producto.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "producto_id": record.get("id") or record.get("producto_id"),
                "nombre": record.get("nombre"),
                "descripcion": record.get("descripcion", ""),
                "precio": float(record.get("precio", 0)),
                "costo": float(record.get("costo", 0)),
                "categoria_id": record.get("categoria_id"),
                "activo": record.get("activo", True),
                "imagen_url": record.get("imagen", "") or record.get("imagen_url", ""),
                "stock": int(record.get("stock", 0)),
                "unidad_medida": record.get("unidad_medida", ""),
                "codigo": record.get("codigo", ""),
                "preparacion_minutos": int(record.get("preparacion_minutos", 0)),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de producto")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_categoria(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de categoría.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "categoria_id": record.get("categoria_id"),
                "nombre": record.get("nombre", ""),
                "descripcion": record.get("descripcion", ""),
                "estado": record.get("estado", True),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de categoría")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_cuenta_mesa(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de cuenta mesa.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Convertir fechas si vienen como strings
            fecha_hora_ini = record.get("fecha_hora_ini")
            if isinstance(fecha_hora_ini, str):
                fecha_hora_ini = datetime.fromisoformat(fecha_hora_ini.replace('Z', '+00:00'))
                
            fecha_hora_fin = record.get("fecha_hora_fin")
            if isinstance(fecha_hora_fin, str):
                fecha_hora_fin = datetime.fromisoformat(fecha_hora_fin.replace('Z', '+00:00'))
            
            # Calcular duración en minutos
            duracion_minutos = None
            if fecha_hora_ini and fecha_hora_fin:
                duracion = fecha_hora_fin - fecha_hora_ini
                duracion_minutos = duracion.total_seconds() / 60
            
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "cuenta_mesa_id": record.get("cuenta_mesa_id"),
                "mesa_id": record.get("mesa_id"),
                "cliente_id": record.get("cliente_id"),
                "usuario_id": record.get("usuario_id"),
                "num_comensales": int(record.get("num_comensales", 0)),
                "fecha_hora_ini": fecha_hora_ini,
                "fecha_hora_fin": fecha_hora_fin,
                "fecha": fecha_hora_ini.date().isoformat() if fecha_hora_ini else None,
                "hora_inicio": fecha_hora_ini.time().isoformat() if fecha_hora_ini else None,
                "dia_semana": fecha_hora_ini.weekday() if fecha_hora_ini else None,
                "duracion_minutos": duracion_minutos,
                "estado": record.get("estado", ""),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de cuenta mesa")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_cliente(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de cliente.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Convertir fechas si vienen como strings o date objects
            fecha_nac = record.get("fecha_nac")
            if isinstance(fecha_nac, str):
                try:
                    fecha_nac = datetime.fromisoformat(fecha_nac.replace('Z', '+00:00'))
                    fecha_nac_iso = fecha_nac.isoformat()
                except:
                    fecha_nac_iso = None
            elif hasattr(fecha_nac, 'isoformat'):  # Más seguro que isinstance(fecha_nac, datetime.date)
                # Convertir date a string ISO para que MongoDB pueda almacenarlo
                fecha_nac_iso = fecha_nac.isoformat()
            else:
                fecha_nac_iso = None
            
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "cliente_id": record.get("cliente_id"),
                "nombre": record.get("nombre", ""),
                "apellido": record.get("apellido", ""),
                "nombre_completo": f"{record.get('nombre', '')} {record.get('apellido', '')}".strip(),
                "nit": record.get("nit", ""),
                "email": record.get("email", ""),
                "pais": record.get("pais", ""),
                "fecha_nac": fecha_nac_iso,  # Usar la versión string
                "direccion": record.get("direccion", ""),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de cliente")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_mesa(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de mesa.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "mesa_id": record.get("mesa_id"),
                "numero": int(record.get("numero", 0)),
                "capacidad_min": int(record.get("capacidad_min", 0)),
                "capacidad_max": int(record.get("capacidad_max", 0)),
                "estado": record.get("estado", True),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de mesa")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_reserva(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de reserva.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Convertir fechas si vienen como strings o date objects
            fecha_reserva = record.get("fecha_reserva")
            if isinstance(fecha_reserva, str):
                try:
                    fecha_reserva = datetime.fromisoformat(fecha_reserva.replace('Z', '+00:00'))
                except:
                    pass
            
            # Convertir datetime.date a string ISO y obtener día de la semana
            fecha_reserva_iso = None
            dia_semana = None
            
            if hasattr(fecha_reserva, 'isoformat'):
                fecha_reserva_iso = fecha_reserva.isoformat()
                if hasattr(fecha_reserva, 'weekday'):
                    dia_semana = fecha_reserva.weekday()
            
            # Procesar hora_reserva
            hora_reserva_str = None  # Asegurarnos de que sea string al final
            hora_reserva = record.get("hora_reserva")
            
            if isinstance(hora_reserva, str):
                # Ya es string, pero aseguremos que tenga formato correcto
                try:
                    hora_parts = hora_reserva.split(":")
                    hora_obj = datetime.now().replace(
                        hour=int(hora_parts[0]), 
                        minute=int(hora_parts[1]), 
                        second=int(hora_parts[2]) if len(hora_parts) > 2 else 0
                    ).time()
                    hora_reserva_str = hora_obj.isoformat()
                except:
                    hora_reserva_str = hora_reserva  # Mantener el string original si no podemos parsearlo
            elif hasattr(hora_reserva, 'isoformat'):
                # Es un objeto time, convertirlo a string
                hora_reserva_str = hora_reserva.isoformat()
            
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "reserva_id": record.get("reserva_id"),
                "mesa_id": record.get("mesa_id"),
                "cliente_id": record.get("cliente_id"),
                "fecha_reserva": fecha_reserva_iso,
                "hora_reserva": hora_reserva_str,  # Usar siempre la versión string
                "dia_semana": dia_semana,
                "cantidad_personas": int(record.get("cantidad_personas", 0)),
                "estado": record.get("estado", ""),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de reserva")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def transform_venta(self, tenant_id: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma registros de venta.
        
        Args:
            tenant_id: ID del tenant
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        transformed = []
        
        for record in records:
            # Crear nuevo documento con transformaciones
            transformed_record = {
                "tenant_id": tenant_id,
                "venta_id": record.get("venta_id"),
                "cuenta_mesa_id": record.get("cuenta_mesa_id"),
                "total": float(record.get("total", 0)),
                "descuento": float(record.get("descuento", 0)),
                "total_neto": float(record.get("total", 0)) - float(record.get("descuento", 0)),
                "estado": record.get("estado", ""),
                "etl_timestamp": now()
            }
            
            transformed.append(transformed_record)
        
        logger.info(f"Transformados {len(transformed)} registros de venta")
        return transformed
    
    @timing_decorator
    @tenant_aware
    def batch_transform(self, tenant_id: int, table_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforma un lote de registros según el tipo de tabla.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            records: Registros a transformar
            
        Returns:
            Registros transformados
        """
        try:
            # Obtener método de transformación específico para la tabla
            transform_method = getattr(self, f"transform_{table_name}", None)
            
            if transform_method is None:
                logger.warning(f"No hay método de transformación específico para {table_name}")
                # Transformación genérica: agregar tenant_id y timestamp
                return [
                    {
                        **record,
                        "tenant_id": tenant_id,
                        "etl_timestamp": now()
                    }
                    for record in records
                ]
            
            # Llamar al método de transformación específico
            # Importante: pasamos tenant_id como argumento nombrado para evitar duplicación
            transformed_records = transform_method(records=records, tenant_id=tenant_id)
            
            return transformed_records
        except Exception as e:
            logger.error(f"Error transformando registros de {table_name}: {str(e)}")
            raise

# Singleton
data_transformer = DataTransformer()

# Función para obtener el transformador de datos
def get_data_transformer():
    return data_transformer