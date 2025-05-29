from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from utils import logger, timing_decorator, tenant_aware, now
from etl.extract import get_postgres_extractor
from etl.transform import get_data_transformer
from etl.load import get_data_loader
from config import get_config

class DataSynchronizer:
    """Sincronizador de datos entre PostgreSQL y MongoDB."""
    
    def __init__(self):
        """Inicializa el sincronizador de datos."""
        self.config = get_config()
        self.extractor = get_postgres_extractor()
        self.transformer = get_data_transformer()
        self.loader = get_data_loader()
    
    @timing_decorator
    @tenant_aware
    def sync_table(self, tenant_id: int, table_name: str, 
                 force_full: bool = False) -> Dict[str, Any]:
        """Sincroniza una tabla desde PostgreSQL a MongoDB.
        
        Args:
            tenant_id: ID del tenant
            table_name: Nombre de la tabla
            force_full: Si se debe forzar sincronización completa
            
        Returns:
            Resultado de la sincronización
        """
        try:
            # Verificar si la tabla está en la lista de sincronización
            if table_name not in self.config.etl.tables_to_sync:
                logger.warning(f"Tabla {table_name} no está configurada para sincronización")
                return {"status": "skipped", "reason": "table_not_configured"}
            
            logger.info(f"Iniciando sincronización de {table_name} para tenant {tenant_id}")
            
            # Obtener estado de última sincronización
            last_sync = None
            if not force_full:
                sync_status = self.loader.get_sync_status(tenant_id=tenant_id, table_name=table_name)
                # Uso de comparaciones explícitas con None
                if sync_status is not None and sync_status.get("last_sync") is not None:
                    last_sync = sync_status.get("last_sync")
                    logger.info(f"Última sincronización: {last_sync}")
            
            # Extraer datos (incremental o completo)
            start_time = now()
            self.loader.update_sync_status(
                tenant_id=tenant_id, table_name=table_name, records_synced=0, status="running"
            )
            
            # Extraer datos
            records = self.extractor.extract_table(
                tenant_id=tenant_id, table_name=table_name, last_sync=last_sync
            )
            
            # Evaluación explícita en lugar de booleana implícita
            if records is None or len(records) == 0:
                logger.info(f"No hay registros nuevos para {table_name}")
                self.loader.update_sync_status(
                    tenant_id=tenant_id, table_name=table_name, records_synced=0, status="completed"
                )
                return {"status": "success", "records": 0}
            
            # Transformar datos
            transformed_records = self.transformer.batch_transform(
                tenant_id=tenant_id, table_name=table_name, records=records
            )
            
            # Cargar datos
            load_result = self.loader.load_records(
                tenant_id=tenant_id, table_name=table_name, records=transformed_records
            )
            
            # Actualizar estado
            records_count = len(transformed_records)
            self.loader.update_sync_status(
                tenant_id=tenant_id, table_name=table_name, records_synced=records_count, status="completed"
            )
            
            end_time = now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success",
                "table": table_name,
                "tenant_id": tenant_id,
                "records": records_count,
                "inserted": load_result.get("inserted", 0),
                "updated": load_result.get("updated", 0),
                "duration_seconds": duration
            }
            
            logger.info(f"Sincronización de {table_name} completada: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error sincronizando {table_name}: {str(e)}")
            
            # Registrar error en estado de sincronización
            self.loader.update_sync_status(
                tenant_id=tenant_id, table_name=table_name, records_synced=0, status="error", error=str(e)
            )
            
            return {
                "status": "error",
                "table": table_name,
                "tenant_id": tenant_id,
                "error": str(e)
            }
    
    @timing_decorator
    @tenant_aware
    def sync_all_tables(self, tenant_id: int, force_full: bool = False) -> List[Dict[str, Any]]:
        """Sincroniza todas las tablas configuradas.
        
        Args:
            tenant_id: ID del tenant
            force_full: Si se debe forzar sincronización completa
            
        Returns:
            Lista de resultados de sincronización
        """
        results = []
        
        for table_name in self.config.etl.tables_to_sync:
            result = self.sync_table(tenant_id=tenant_id, table_name=table_name, force_full=force_full)
            results.append(result)
            
        return results

# Singleton
data_synchronizer = DataSynchronizer()

# Función para obtener el sincronizador de datos
def get_data_synchronizer():
    return data_synchronizer