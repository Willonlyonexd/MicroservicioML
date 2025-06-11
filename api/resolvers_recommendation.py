# resolvers_recommendation.py
import sys
import os
from pathlib import Path

# Asegurar que el módulo recommendation está disponible
ml_path = str(Path(__file__).parent.parent / 'ml')
if ml_path not in sys.path:
    sys.path.append(ml_path)

from ml.recommendation.recommendation_engine import RecommendationEngine

# Inicializar el motor de recomendación (se hace de manera perezosa)
_recommendation_engine = None

def get_recommendation_engine():
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine

async def resolve_personal_recommendations(obj, info, client_id, tenant_id, limit=5, filter=None):
    """Resolver para recomendaciones personalizadas"""
    try:
        engine = get_recommendation_engine()
        
        # Extraer datos del filtro
        exclude_products = filter.get('excludeProductIds', []) if filter else []
        category_ids = filter.get('categoryIds', []) if filter else []
        context = filter.get('context', {}) if filter else {}
        
        # Obtener recomendaciones del motor
        recommendations = engine.get_personal_recommendations(
            client_id=client_id,
            tenant_id=tenant_id,
            limit=limit,
            context=context,
            exclude_products=exclude_products,
            category_ids=category_ids
        )
        
        return recommendations
    except Exception as e:
        print(f"Error en resolver de recomendaciones personalizadas: {str(e)}")
        return []

async def resolve_general_recommendations(obj, info, tenant_id, limit=10, filter=None):
    """Resolver para recomendaciones generales"""
    try:
        engine = get_recommendation_engine()
        
        # Extraer datos del filtro
        category_ids = filter.get('categoryIds', []) if filter else []
        context = filter.get('context', {}) if filter else {}
        
        # Obtener recomendaciones del motor
        recommendations = engine.get_general_recommendations(
            tenant_id=tenant_id,
            limit=limit,
            context=context,
            category_ids=category_ids
        )
        
        return recommendations
    except Exception as e:
        print(f"Error en resolver de recomendaciones generales: {str(e)}")
        return []

# Función para registrar los resolvers en el esquema principal
def register_recommendation_resolvers(schema):
    schema.get_query_type().fields['personalRecommendations'].resolve = resolve_personal_recommendations
    schema.get_query_type().fields['generalRecommendations'].resolve = resolve_general_recommendations
    return schema