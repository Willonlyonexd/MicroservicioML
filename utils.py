import time
import logging
from datetime import datetime
from functools import wraps
from loguru import logger
from typing import Callable, Any, Dict, Optional
from config import get_config

# Configurar logger
config = get_config()
logger.remove()
logger.add(
    config.log.file if config.log.file else "ml_service.log",
    level=config.log.level,
    rotation="500 MB",
    retention="10 days"
)
if config.dev_mode:
    logger.add(lambda msg: print(msg), level="DEBUG", colorize=True)

def setup_logging():
    """Configurar logging para la aplicación"""
    logging.basicConfig(
        level=getattr(logging, config.log.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def timing_decorator(func: Callable) -> Callable:
    """Decorador para medir el tiempo de ejecución de funciones"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Función {func.__name__} ejecutada en {end_time - start_time:.2f} segundos")
        return result
    return wrapper

def get_tenant_id_from_args(args: tuple, kwargs: Dict[str, Any]) -> Optional[int]:
    """Extrae el tenant_id de los argumentos de una función"""
    # Primero buscar en kwargs
    if 'tenant_id' in kwargs:
        return kwargs['tenant_id']
    
    # Buscar en args si hay algún diccionario con tenant_id
    for arg in args:
        if isinstance(arg, dict) and 'tenant_id' in arg:
            return arg['tenant_id']
    
    return None

def tenant_aware(func: Callable) -> Callable:
    """Decorador para funciones que requieren tenant_id"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant_id = kwargs.get('tenant_id')
        
        # Buscar en objetos self si existe un atributo tenant_id
        if tenant_id is None and len(args) > 0 and hasattr(args[0], 'tenant_id'):
            tenant_id = args[0].tenant_id
            
        # Si no hay tenant_id en kwargs ni en self, buscarlo en otros argumentos
        if tenant_id is None:
            for arg in args:
                if isinstance(arg, int):  # Posible tenant_id como argumento posicional
                    tenant_id = arg
                    break
                elif isinstance(arg, dict) and 'tenant_id' in arg:
                    tenant_id = arg['tenant_id']
                    break
        
        if tenant_id is None:
            # Imprimir información de depuración para diagnosticar
            logger.debug(f"No se encontró tenant_id en llamada a {func.__name__}")
            logger.debug(f"Args: {args}")
            logger.debug(f"Kwargs: {kwargs}")
            raise ValueError("tenant_id es requerido para esta operación")
        
        # Asegurar que tenant_id esté siempre en kwargs para facilitar su uso en la función
        kwargs['tenant_id'] = tenant_id
        logger.debug(f"Ejecutando {func.__name__} para tenant_id={tenant_id}")
        return func(*args, **kwargs)
    return wrapper

def format_date(dt: datetime) -> str:
    """Formatea una fecha en formato ISO"""
    return dt.isoformat() if dt else None

def now() -> datetime:
    """Retorna la fecha y hora actual en UTC"""
    return datetime.utcnow()