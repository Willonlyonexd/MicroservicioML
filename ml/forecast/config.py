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
    'epochs': 50,             # Añadido para compatibilidad con model.py
    'batch_size': 32,         # Añadido para compatibilidad con model.py
    'patience': 10,           # Añadido para compatibilidad con model.py
    'optimizer': 'adam',      # Añadido para compatibilidad con model.py
    'loss': 'mse',            # Añadido para compatibilidad con model.py
    'metrics': ['mae']        # Añadido para compatibilidad con model.py
}

# Parámetros de datos
DATA_PARAMS = {
    'sequence_length': 7,     # Días de historia para predecir
    'horizon': 7,             # Días futuros a predecir
    'batch_size': 32,         # Tamaño del batch
    'train_split': 0.8,       # Proporción de datos para entrenamiento
    'features': [             # Features a utilizar
        'day_of_week', 
        'month', 
        'is_weekend', 
        'is_holiday'
    ]
}

# Añadir configuración para productos
PRODUCT_PARAMS = {
    'min_historical_data': 30,  # Mínimo de datos históricos necesarios
    'top_products_limit': 100,  # Límite de productos a predecir (los más vendidos)
    'batch_training': True,     # Entrenar modelos en lotes
    'confidence_interval': 0.9, # Intervalo de confianza para predicciones
    'min_data_points': 30,      # Añadido para compatibilidad con model.py
    'min_sales': 10,            # Añadido para compatibilidad con model.py
    'top_n': 10                 # Añadido para compatibilidad con model.py
}

# Parámetros de entrenamiento
TRAINING_PARAMS = {
    'epochs': 50,             # Épocas máximas
    'patience': 10,           # Early stopping patience
    'validation_split': 0.2,  # Proporción para validación
}

# Parámetros de MongoDB
MONGO_PARAMS = {
    'collection_sales': 'raw_pedidos',
    'collection_details': 'raw_pedido_detalles',
    'collection_predictions': 'ml_predicciones',
    'collection_products': 'raw_productos',
    'collection_product_predictions': 'ml_predicciones_producto',  # Nueva colección
    'collection_forecasts': 'ml_predicciones',                     # Añadido para compatibilidad
    'collection_product_forecasts': 'ml_predicciones_producto'     # Añadido para compatibilidad
}