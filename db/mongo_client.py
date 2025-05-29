from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import get_config
from utils import logger, timing_decorator
from typing import Optional, Dict, Any, List

class MongoManager:
    """Gestor de conexiones a MongoDB con soporte multi-tenant."""
    
    def __init__(self):
        """Inicializa el gestor de MongoDB."""
        self.config = get_config()
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._connect()
    
    @timing_decorator
    def _connect(self) -> None:
        """Establece conexión con MongoDB."""
        try:
            # Conectar a MongoDB con tiempo de espera de 5 segundos
            self.client = MongoClient(
                self.config.database.mongo_uri,
                serverSelectionTimeoutMS=5000
            )
            
            # Verificar conexión
            self.client.admin.command('ping')
            
            # Seleccionar base de datos
            self.db = self.client[self.config.database.mongo_db]
            
            logger.info(f"Conexión a MongoDB establecida: {self.config.database.mongo_uri}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"No se pudo conectar a MongoDB: {str(e)}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """Obtiene una colección de MongoDB.
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Objeto Collection de pymongo
        """
        if self.db is None:  # Comparación explícita con None
            self._connect()
        return self.db[collection_name]
    
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Inserta un documento en una colección.
        
        Args:
            collection_name: Nombre de la colección
            document: Documento a insertar
            
        Returns:
            ID del documento insertado
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Inserta múltiples documentos en una colección.
        
        Args:
            collection_name: Nombre de la colección
            documents: Lista de documentos a insertar
            
        Returns:
            Lista de IDs de documentos insertados
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Busca un documento en una colección.
        
        Args:
            collection_name: Nombre de la colección
            query: Consulta de búsqueda
            
        Returns:
            Documento encontrado o None
        """
        collection = self.get_collection(collection_name)
        return collection.find_one(query)
    
    def find(self, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca documentos en una colección.
        
        Args:
            collection_name: Nombre de la colección
            query: Consulta de búsqueda
            
        Returns:
            Lista de documentos encontrados
        """
        collection = self.get_collection(collection_name)
        return list(collection.find(query))
    
    def update_one(self, collection_name: str, query: Dict[str, Any], 
                  update: Dict[str, Any], upsert: bool = False) -> int:
        """Actualiza un documento en una colección.
        
        Args:
            collection_name: Nombre de la colección
            query: Consulta para encontrar el documento
            update: Actualización a realizar
            upsert: Si se debe insertar si no existe
            
        Returns:
            Número de documentos modificados
        """
        collection = self.get_collection(collection_name)
        result = collection.update_one(query, update, upsert=upsert)
        return result.modified_count
    
    def upsert_many(self, collection_name: str, documents: List[Dict[str, Any]], 
                   key_field: str) -> Dict[str, int]:
        """Inserta o actualiza múltiples documentos basado en un campo clave.
        
        Args:
            collection_name: Nombre de la colección
            documents: Lista de documentos
            key_field: Campo a usar como clave para determinar si actualizar
            
        Returns:
            Diccionario con conteo de operaciones
        """
        collection = self.get_collection(collection_name)
        inserted = 0
        updated = 0
        
        for doc in documents:
            if key_field not in doc:
                logger.warning(f"Campo clave '{key_field}' no encontrado en documento: {doc}")
                continue
                
            result = collection.update_one(
                {key_field: doc[key_field]},
                {"$set": doc},
                upsert=True
            )
            
            # Usar comparación explícita con cero
            if result.matched_count > 0:
                updated += 1
            else:
                inserted += 1
        
        return {"inserted": inserted, "updated": updated}
    
    def create_index(self, collection_name: str, keys: Dict[str, int], **kwargs) -> str:
        """Crea un índice en una colección.
        
        Args:
            collection_name: Nombre de la colección
            keys: Campos del índice (1 ascendente, -1 descendente)
            **kwargs: Argumentos adicionales para el índice
            
        Returns:
            Nombre del índice creado
        """
        collection = self.get_collection(collection_name)
        result = collection.create_index(keys, **kwargs)
        logger.info(f"Índice '{result}' creado en {collection_name}")
        return result
    
    def tenant_query(self, tenant_id: int, additional_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """Genera una consulta que incluye el tenant_id.
        
        Args:
            tenant_id: ID del tenant
            additional_query: Consulta adicional a combinar
            
        Returns:
            Consulta combinada con tenant_id
        """
        query = {"tenant_id": tenant_id}
        if additional_query is not None:  # Comparación explícita con None
            query.update(additional_query)
        return query
    
    def close(self) -> None:
        """Cierra la conexión con MongoDB."""
        if self.client is not None:  # Comparación explícita con None
            self.client.close()
            logger.info("Conexión a MongoDB cerrada")
            self.client = None
            self.db = None

# Singleton
mongo_manager = MongoManager()

# Función para obtener el gestor de MongoDB
def get_mongo_manager() -> MongoManager:
    return mongo_manager