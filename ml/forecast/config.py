"""
Configuración para el módulo de forecasting.
"""

# Parámetros del modelo
MODEL_PARAMS = {
    'units': 64,              # Número de neuronas en capa LSTM
    'lstm_units': 64,         # Alias para compatibilidad con el código
    'stack_layers': 2,        # Capas LSTM apiladas
    'dropout': 0.2,           # Dropout para regularización
    'dropout_rate': 0.2,      # Alias para compatibilidad con el código
    'learning_rate': 0.001,   # Tasa de aprendizaje
    'epochs': 50,             # Épocas para entrenamiento
    'batch_size': 32,         # Tamaño de batch
    'patience': 10,           # Paciencia para early stopping
    'optimizer': 'adam',      # Optimizador
    'loss': 'mse',            # Función de pérdida
    'metrics': ['mae']        # Métricas de evaluación
}

# Parámetros de datos
DATA_PARAMS = {
    'sequence_length': 200,     # Días de historia para predecir
    'horizon': 30,             # Días futuros a predecir
    'batch_size': 32,         # Tamaño del batch
    'train_split': 0.6,       # Proporción de datos para entrenamiento
    'features': [             # Features a utilizar
        'day_of_week', 
        'month', 
        'is_weekend', 
        'is_holiday'
    ]
}

# Parámetros para productos
PRODUCT_PARAMS = {
    'min_historical_data': 200   ,  # Mínimo de datos históricos necesarios
    'top_products_limit': 29,   # Límite de productos a predecir (los más vendidos)
    'batch_training': True,     # Entrenar modelos en lotes
    'confidence_interval': 0.6, # Intervalo de confianza para predicciones
    'min_data_points': 10,      # Mínimo de puntos de datos necesarios
    'min_sales': 20,            # Ventas mínimas para considerar producto
    'top_n': 10                 # Número de productos top por defecto
}

# Parámetros para agregación
AGGREGATION_PARAMS = {
    'max_forecast_days': 200,    # Máximo de días para predicción 
    'max_forecast_weeks': 28,    # Máximo de semanas para predicción
    'max_forecast_months': 20    # Máximo de meses para predicción
}

# Parámetros de entrenamiento
TRAINING_PARAMS = {
    'epochs': 50,             # Épocas máximas
    'patience': 10,           # Early stopping patience
    'validation_split': 0.2,  # Proporción para validación
}

# Parámetros de MongoDB
MONGO_PARAMS = {
    'collection_sales': 'raw_pedido',           # Cambiado a raw_pedido como principal
    'collection_details': 'raw_pedido_detalle',
    'collection_predictions': 'ml_predicciones',
    'collection_products': 'raw_producto',
    'collection_product_predictions': 'ml_predicciones_producto',
    'collection_category_predictions': 'ml_predicciones_categoria',  # Nueva colección
    'collection_forecasts': 'ml_predicciones',
    'collection_product_forecasts': 'ml_predicciones_producto'
}