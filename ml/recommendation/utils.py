import numpy as np
from datetime import datetime, timedelta
import logging

def get_time_context(current_datetime=None):
    """
    Obtiene información contextual de tiempo
    
    Args:
        current_datetime: Fecha y hora actual, o None para usar la actual del sistema
    """
    # Si no se proporciona fecha/hora, usar la referencia de 2025-06-11
    if current_datetime is None:
        current_datetime = datetime(2025, 6, 11, 10, 22, 19)
    
    hour = current_datetime.hour
    
    if 6 <= hour < 12:
        time_of_day = "MORNING"
    elif 12 <= hour < 15:
        time_of_day = "LUNCH"
    elif 15 <= hour < 18:
        time_of_day = "AFTERNOON"
    else:
        time_of_day = "DINNER"
    
    day_of_week = current_datetime.weekday()  # 0=Lunes, 6=Domingo
    
    # Determinar temporada (para el hemisferio sur)
    month = current_datetime.month
    if month in [12, 1, 2]:
        season = "SUMMER"
    elif month in [3, 4, 5]:
        season = "FALL"
    elif month in [6, 7, 8]:
        season = "WINTER"
    else:
        season = "SPRING"
    
    return {
        "timeOfDay": time_of_day,
        "dayOfWeek": day_of_week,
        "season": season,
        "datetime": current_datetime
    }

def log_recommendation_request(client_id, tenant_id, recommendations):
    """Registra las solicitudes de recomendación para análisis"""
    logger = logging.getLogger('recommendation_logs')
    
    try:
        # Registrar datos básicos
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "tenant_id": tenant_id,
            "num_recommendations": len(recommendations)
        }
        
        # Registrar IDs de productos recomendados
        product_ids = [rec["product"]["id"] for rec in recommendations]
        log_data["product_ids"] = product_ids
        
        logger.info(f"RECOMMENDATION_LOG: {log_data}")
    except Exception as e:
        logger.error(f"Error al registrar recomendación: {str(e)}")