from .model import RecommendationModel
from .data_manager import RecommendationDataManager
from .utils import get_time_context, log_recommendation_request
from sqlalchemy import create_engine
import logging
import os

class RecommendationEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecommendationEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('recommendation_engine')
        
        # Inicializar conexión a BD usando la URI proporcionada
        db_uri = os.getenv('POSTGRES_URI', 'postgresql://db_erpfinal_user:EqEkEukceSrqrTHpuo4X0T8ZD79GVtUB@dpg-d133na15pdvs73dad96g-a.oregon-postgres.render.com:5432/db_erpfinal?sslmode=require')
        self.db_engine = create_engine(db_uri)
        
        # Inicializar componentes
        try:
            self.data_manager = RecommendationDataManager(self.db_engine)
            self.model = RecommendationModel(self.data_manager)
            
            self.logger.info("Sistema de recomendación inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar el sistema de recomendación: {str(e)}")
        
        self._initialized = True
    
    def get_personal_recommendations(self, client_id, tenant_id, limit=5, context=None, exclude_products=None, category_ids=None):
        """
        Obtiene recomendaciones personalizadas para un cliente.
        
        Args:
            client_id: ID del cliente
            tenant_id: ID del tenant (restaurante)
            limit: Número máximo de recomendaciones
            context: Diccionario con información contextual
            exclude_products: Lista de IDs de productos a excluir
            category_ids: Lista de IDs de categorías para filtrar
            
        Returns:
            Lista de recomendaciones personalizadas
        """
        try:
            self.logger.info(f"Generando recomendaciones personalizadas para cliente {client_id} (tenant {tenant_id})")
            
            # Si no se proporciona contexto, usar el actual
            if not context:
                context = get_time_context()
            
            # Obtener recomendaciones del modelo
            recommendations = self.model.get_personal_recommendations(
                client_id=client_id,
                tenant_id=tenant_id,
                limit=limit,
                exclude_products=exclude_products,
                category_ids=category_ids,
                context=context
            )
            
            # Registrar recomendación para análisis
            log_recommendation_request(client_id, tenant_id, recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones personalizadas: {str(e)}")
            return []
    
    def get_general_recommendations(self, tenant_id, limit=5, context=None, category_ids=None):
        """
        Obtiene recomendaciones generales basadas en popularidad.
        
        Args:
            tenant_id: ID del tenant (restaurante)
            limit: Número máximo de recomendaciones
            context: Diccionario con información contextual
            category_ids: Lista de IDs de categorías para filtrar
            
        Returns:
            Lista de recomendaciones generales
        """
        try:
            self.logger.info(f"Generando recomendaciones generales para tenant {tenant_id}")
            
            # Si no se proporciona contexto, usar el actual
            if not context:
                context = get_time_context()
            
            # Obtener recomendaciones del modelo
            recommendations = self.model.get_general_recommendations(
                tenant_id=tenant_id,
                limit=limit,
                context=context,
                category_ids=category_ids
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones generales: {str(e)}")
            return []