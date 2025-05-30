import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import json

from .data_fixed import TimeSeriesDataProcessor
from .config import MODEL_PARAMS, DATA_PARAMS, MONGO_PARAMS, AGGREGATION_PARAMS

class TFForecaster:
    """
    Modelo de forecasting basado en TensorFlow para predicción de series temporales.
    """
    
    def __init__(self, db_client=None):
        """
        Inicializa el modelo de forecasting.
        
        Args:
            db_client: Cliente de base de datos MongoDB
        """
        self.model = None
        self.data_processor = TimeSeriesDataProcessor(db_client)
        self.db_client = db_client
        self.product_models = {}
        
    def build_model(self, input_shape):
        """
        Construye la arquitectura del modelo LSTM.
        
        Args:
            input_shape: Forma de los datos de entrada (sequence_length, features)
            
        Returns:
            model: Modelo compilado de Keras
        """
        model = Sequential([
            LSTM(MODEL_PARAMS['lstm_units'], 
                 activation='relu', 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(MODEL_PARAMS['dropout_rate']),
            LSTM(MODEL_PARAMS['lstm_units'] // 2, 
                 activation='relu'),
            Dropout(MODEL_PARAMS['dropout_rate']),
            Dense(DATA_PARAMS['horizon'])
        ])
        
        model.compile(
            optimizer=MODEL_PARAMS['optimizer'],
            loss=MODEL_PARAMS['loss'],
            metrics=MODEL_PARAMS['metrics']
        )
        
        return model
    
    def train(self, data=None, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size']):
        """
        Entrena el modelo con datos históricos.
        
        Args:
            data: DataFrame con datos históricos (opcional, si no se proporciona se extraen de la BD)
            epochs: Número de épocas para entrenar
            batch_size: Tamaño del batch
            
        Returns:
            history: Historial de entrenamiento
        """
        # Si no se proporcionan datos, extraerlos
        if data is None:
            data = self.data_processor.fetch_historical_data()
            
        # Agregar características temporales
        data_with_features = self.data_processor.add_time_features(data)
        
        # Crear secuencias
        X_train, y_train, X_val, y_val = self.data_processor.create_sequences(data_with_features)
        
        if len(X_train) == 0:
            raise ValueError("No hay suficientes datos para entrenar el modelo. Se requieren al menos 14 días de datos históricos.")
        
        # Construir modelo
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Early stopping para evitar overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=MODEL_PARAMS['patience'],
            restore_best_weights=True
        )
        
        # Configurar callbacks
        callbacks = [early_stopping]
        
        # Entrenar modelo
        if len(X_val) > 0:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Si no hay datos de validación, entrenar sin ellos
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def train_product_models(self, product_ids=None, top_n=10, epochs=MODEL_PARAMS['epochs']):
        """
        Entrena modelos específicos para productos seleccionados.
        
        Args:
            product_ids: Lista de IDs de productos a modelar (opcional)
            top_n: Si no se proporciona product_ids, entrenar para los top N productos
            epochs: Número de épocas para entrenar
            
        Returns:
            dict: Resultados del entrenamiento por producto
        """
        # Si no se proporcionan IDs específicos, obtener los top N productos
        if product_ids is None:
            product_ids = self.data_processor.get_top_products(limit=top_n)
            
        # Si aún no hay IDs, no podemos entrenar modelos de productos
        if not product_ids:
            logging.warning("No se pudieron identificar productos para entrenamiento específico")
            return {}
            
        # Obtener datos históricos por producto
        product_data = self.data_processor.fetch_product_historical_data(top_n=top_n)
        
        # Resultados de entrenamiento
        training_results = {}
        
        # Entrenar un modelo para cada producto
        for product_id in product_ids:
            try:
                logging.info(f"Entrenando modelo para producto {product_id}...")
                
                # Filtrar datos para este producto y añadir características temporales
                product_df = product_data[product_data['producto_id'] == product_id].copy()
                
                # Verificar si hay suficientes datos
                if len(product_df) < DATA_PARAMS['sequence_length'] + DATA_PARAMS['horizon'] + 5:
                    logging.warning(f"Datos insuficientes para producto {product_id}. Omitiendo.")
                    training_results[product_id] = {
                        'status': 'skipped',
                        'reason': 'insufficient_data'
                    }
                    continue
                
                # Añadir características temporales
                product_df_features = self.data_processor.add_time_features(product_df)
                
                # Crear secuencias para este producto
                X_train, y_train, X_val, y_val = self.data_processor.create_product_sequences(
                    product_df_features, product_id)
                
                # Si no se pudieron crear secuencias, continuar con el siguiente producto
                if X_train is None:
                    logging.warning(f"No se pudieron crear secuencias para producto {product_id}. Omitiendo.")
                    training_results[product_id] = {
                        'status': 'skipped',
                        'reason': 'sequence_creation_failed'
                    }
                    continue
                
                # Construir modelo para este producto
                product_model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
                # Early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=MODEL_PARAMS['patience'],
                    restore_best_weights=True
                )
                
                # Entrenar modelo
                if len(X_val) > 0:
                    history = product_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=min(32, len(X_train)),  # Ajustar batch_size según datos disponibles
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                else:
                    # Entrenar sin validación si no hay datos suficientes
                    history = product_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=min(32, len(X_train)),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                
                # Guardar modelo en memoria
                self.product_models[product_id] = product_model
                
                # Guardar resultados
                training_results[product_id] = {
                    'status': 'success',
                    'val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                    'epochs': len(history.history['loss'])
                }
                
                logging.info(f"Modelo para producto {product_id} entrenado exitosamente.")
                
            except Exception as e:
                logging.error(f"Error al entrenar modelo para producto {product_id}: {str(e)}")
                training_results[product_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return training_results
    
    def predict_next_days(self, days=DATA_PARAMS['horizon']):
        """
        Predice ventas para los próximos días.
        
        Args:
            days: Número de días a predecir (debe ser <= horizon del modelo)
            
        Returns:
            dict: Predicciones para los próximos días
        """
        # Verificar que el modelo esté entrenado
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones.")
            
        # Obtener datos históricos
        data = self.data_processor.fetch_historical_data()
        
        # Agregar características temporales
        data_with_features = self.data_processor.add_time_features(data)
        
        # Preparar última secuencia
        last_sequence = self.data_processor.prepare_last_sequence(data_with_features)
        
        # Hacer predicción
        scaled_prediction = self.model.predict(last_sequence)
        
        # Invertir normalización
        prediction = self.data_processor.inverse_transform(scaled_prediction[0])
        
        # Asegurar que no predecimos más días de los solicitados
        prediction = prediction[:days]
        
        # Crear fechas futuras
        last_date = data['fecha'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(len(prediction))]
        
        # Formato de salida
        result = []
        for i, (date, pred) in enumerate(zip(future_dates, prediction)):
            result.append({
                'fecha': date.strftime('%Y-%m-%d'),
                'dia': i + 1,
                'prediccion': float(max(0, pred)),  # Asegurar valores no negativos
                'confianza': 0.9 - (i * 0.05)  # La confianza disminuye con días más lejanos
            })
        
        return result
    
    def predict_aggregated(self, period='day', horizon=None):
        """
        Predice ventas agregadas por día, semana o mes.
        
        Args:
            period: 'day', 'week' o 'month'
            horizon: Número de períodos a predecir (si es None, usar configuración por defecto)
        
        Returns:
            list: Predicciones para los períodos solicitados
        """
        # Determinar número de días base a predecir
        days_needed = {
            'day': DATA_PARAMS['horizon'],
            'week': AGGREGATION_PARAMS.get('max_forecast_weeks', 8) * 7,
            'month': AGGREGATION_PARAMS.get('max_forecast_months', 3) * 30
        }.get(period, DATA_PARAMS['horizon'])
        
        if horizon:
            if period == 'week':
                days_needed = horizon * 7
            elif period == 'month':
                days_needed = horizon * 30
            else:
                days_needed = horizon
        
        # Asegurar que no excedemos el límite del modelo
        max_forecast = AGGREGATION_PARAMS.get('max_forecast_days', 60)
        days_needed = min(days_needed, max_forecast)
        
        # Obtener predicciones diarias
        daily_predictions = self.predict_next_days(days=days_needed)
        
        if period == 'day':
            return daily_predictions
        
        # Para agregaciones semanales o mensuales
        result = []
        
        if period == 'week':
            # Agrupar por semanas
            week_data = {}
            current_week = 1
            days_in_week = 0
            week_total = 0
            week_start = None
            
            for pred in daily_predictions:
                date = datetime.strptime(pred['fecha'], '%Y-%m-%d')
                
                if week_start is None:
                    week_start = date
                
                week_total += pred['prediccion']
                days_in_week += 1
                
                if days_in_week == 7:
                    # Completamos una semana
                    result.append({
                        'periodo': current_week,
                        'fecha_inicio': week_start.strftime('%Y-%m-%d'),
                        'fecha_fin': pred['fecha'],
                        'prediccion': week_total,
                        'confianza': 0.85 - (current_week * 0.05)  # Reducir confianza con el tiempo
                    })
                    
                    current_week += 1
                    days_in_week = 0
                    week_total = 0
                    week_start = None
            
            # Añadir última semana parcial si existe
            if days_in_week > 0:
                result.append({
                    'periodo': current_week,
                    'fecha_inicio': week_start.strftime('%Y-%m-%d'),
                    'fecha_fin': daily_predictions[-1]['fecha'],
                    'prediccion': week_total,
                    'confianza': 0.85 - (current_week * 0.05)
                })
        
        elif period == 'month':
            # Agrupar por meses
            month_data = {}
            
            for pred in daily_predictions:
                date = datetime.strptime(pred['fecha'], '%Y-%m-%d')
                month_key = f"{date.year}-{date.month:02d}"
                
                if month_key not in month_data:
                    month_data[month_key] = {
                        'total': 0,
                        'days': 0,
                        'start_date': date,
                        'end_date': date
                    }
                
                month_data[month_key]['total'] += pred['prediccion']
                month_data[month_key]['days'] += 1
                month_data[month_key]['end_date'] = date
            
            # Convertir a formato de resultado
            for i, (month_key, data) in enumerate(month_data.items()):
                result.append({
                    'periodo': i + 1,
                    'fecha_inicio': data['start_date'].strftime('%Y-%m-%d'),
                    'fecha_fin': data['end_date'].strftime('%Y-%m-%d'),
                    'prediccion': data['total'],
                    'confianza': 0.8 - (i * 0.1)  # Reducir confianza con el tiempo
                })
        
        return result
    
    def predict_product_demand(self, product_id=None, days=DATA_PARAMS['horizon']):
        """
        Predice demanda por producto para los próximos días.
        
        Args:
            product_id: ID del producto específico (opcional)
            days: Número de días a predecir
            
        Returns:
            pd.DataFrame: Predicciones por producto para los próximos días
        """
        # Si no se especifica producto, predecir para todos los productos con modelo
        if product_id is None:
            if not self.product_models:
                logging.warning("No hay modelos de productos entrenados.")
                return pd.DataFrame()
                
            all_predictions = []
            
            # Obtener datos históricos por producto
            product_data = self.data_processor.fetch_product_historical_data()
            
            # Agregar características temporales
            product_data_features = self.data_processor.add_time_features(product_data)
            
            # Predecir para cada producto
            for prod_id, prod_model in self.product_models.items():
                try:
                    # Preparar última secuencia para este producto
                    last_sequence = self.data_processor.prepare_last_sequence(
                        product_data_features, product_id=prod_id)
                    
                    if last_sequence is None:
                        logging.warning(f"No se pudo preparar secuencia para producto {prod_id}. Omitiendo.")
                        continue
                    
                    # Hacer predicción
                    scaled_prediction = prod_model.predict(last_sequence)
                    
                    # Invertir normalización
                    prediction = self.data_processor.inverse_transform(
                        scaled_prediction[0], product_id=prod_id)
                    
                    # Asegurar que no predecimos más días de los solicitados
                    prediction = prediction[:days]
                    
                    # Crear fechas futuras
                    product_df = product_data[product_data['producto_id'] == prod_id]
                    if not product_df.empty:
                        last_date = product_df['fecha'].max()
                        future_dates = [last_date + timedelta(days=i+1) for i in range(len(prediction))]
                        
                        # Obtener información del producto
                        product_info = self.data_processor.get_product_info(prod_id)
                        
                        # Agregar predicciones a la lista
                        for i, (date, pred) in enumerate(zip(future_dates, prediction)):
                            all_predictions.append({
                                'fecha': date,
                                'dia': i + 1,
                                'producto_id': prod_id,
                                'nombre_producto': product_info.get('nombre', f'Producto {prod_id}'),
                                'categoria_id': product_info.get('categoria_id', None),
                                'prediccion': float(max(0, pred)),  # Asegurar valores no negativos
                                'confianza': 0.9 - (i * 0.05)  # La confianza disminuye con días más lejanos
                            })
                except Exception as e:
                    logging.error(f"Error al predecir para producto {prod_id}: {str(e)}")
            
            # Convertir a DataFrame
            if all_predictions:
                return pd.DataFrame(all_predictions)
            else:
                return pd.DataFrame()
        else:
            # Predecir para un producto específico
            if product_id not in self.product_models:
                logging.warning(f"No hay modelo entrenado para producto {product_id}.")
                return pd.DataFrame()
                
            # Obtener datos históricos para este producto
            product_data = self.data_processor.fetch_product_historical_data(product_id=product_id)
            
            # Agregar características temporales
            product_data_features = self.data_processor.add_time_features(product_data)
            
            # Preparar última secuencia
            last_sequence = self.data_processor.prepare_last_sequence(
                product_data_features, product_id=product_id)
            
            if last_sequence is None:
                logging.warning(f"No se pudo preparar secuencia para producto {product_id}.")
                return pd.DataFrame()
            
            # Hacer predicción
            scaled_prediction = self.product_models[product_id].predict(last_sequence)
            
            # Invertir normalización
            prediction = self.data_processor.inverse_transform(
                scaled_prediction[0], product_id=product_id)
            
            # Asegurar que no predecimos más días de los solicitados
            prediction = prediction[:days]
            
            # Crear fechas futuras
            last_date = product_data['fecha'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(len(prediction))]
            
            # Obtener información del producto
            product_info = self.data_processor.get_product_info(product_id)
            
            # Preparar resultados
            results = []
            for i, (date, pred) in enumerate(zip(future_dates, prediction)):
                results.append({
                    'fecha': date,
                    'dia': i + 1,
                    'producto_id': product_id,
                    'nombre_producto': product_info.get('nombre', f'Producto {product_id}'),
                    'categoria_id': product_info.get('categoria_id', None),
                    'prediccion': float(max(0, pred)),  # Asegurar valores no negativos
                    'confianza': 0.9 - (i * 0.05)  # La confianza disminuye con días más lejanos
                })
            
            return pd.DataFrame(results)
    
    def predict_category_demand(self, category_id=None, days=DATA_PARAMS['horizon']):
        """
        Predice demanda por categoría para los próximos días.
        
        Args:
            category_id: ID de la categoría específica (opcional)
            days: Número de días a predecir
            
        Returns:
            list: Predicciones por categoría para los próximos días
        """
        # Obtener predicciones por producto
        product_predictions = self.predict_product_demand(days=days)
        
        if product_predictions.empty:
            logging.warning("No hay predicciones por producto disponibles para agregar por categoría.")
            return []
        
        # Verificar si tenemos categoría_id en las predicciones
        if 'categoria_id' not in product_predictions.columns:
            logging.warning("Las predicciones de productos no contienen información de categoría.")
            return []
        
        # Si se especificó una categoría, filtrar predicciones
        if category_id is not None:
            product_predictions = product_predictions[product_predictions['categoria_id'] == category_id]
            
            if product_predictions.empty:
                logging.warning(f"No hay productos de la categoría {category_id} con predicciones disponibles.")
                return []
        
        # Agrupar predicciones por fecha y categoría
        result = []
        
        # Convertir DataFrame a diccionario para facilitar manipulación
        pred_dict = {}
        for _, row in product_predictions.iterrows():
            date_str = row['fecha'].strftime('%Y-%m-%d') if isinstance(row['fecha'], datetime) else row['fecha']
            category = row['categoria_id']
            
            # Omitir productos sin categoría
            if pd.isna(category):
                continue
            
            key = (date_str, category)
            if key not in pred_dict:
                # Obtener información de la categoría
                category_info = self.data_processor.get_category_info(category)
                category_name = category_info.get('nombre', f'Categoría {category}')
                
                pred_dict[key] = {
                    'fecha': date_str,
                    'dia': row['dia'],
                    'categoria_id': category,
                    'nombre_categoria': category_name,
                    'prediccion': 0,
                    'productos': 0,
                    'confianza': 0
                }
            
            pred_dict[key]['prediccion'] += row['prediccion']
            pred_dict[key]['productos'] += 1
            pred_dict[key]['confianza'] += row['confianza']
        
        # Promediar confianza y formatear resultados
        for key, data in pred_dict.items():
            if data['productos'] > 0:
                data['confianza'] /= data['productos']  # Promedio de confianzas
            result.append(data)
        
        # Ordenar por fecha y categoría
        result.sort(key=lambda x: (x['dia'], x['categoria_id']))
        
        return result
    
    def save_model(self, path="models/forecast"):
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar.")
            
        # Crear directorio si no existe
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelo
        self.model.save(os.path.join(path, "tf_forecaster.h5"))
        
        # Guardar parámetros de normalización
        scaler_params = {
            'mean': self.data_processor.mean.to_dict(),
            'std': self.data_processor.std.to_dict()
        }
        
        with open(os.path.join(path, "scaler_params.json"), 'w') as f:
            json.dump(scaler_params, f)
            
        logging.info(f"Modelo guardado en {path}")
    
    def save_product_models(self, path="models/forecast/products"):
        """
        Guarda los modelos de productos entrenados.
        
        Args:
            path: Ruta base donde guardar los modelos
        """
        if not self.product_models:
            raise ValueError("No hay modelos de productos para guardar.")
            
        # Crear directorio si no existe
        os.makedirs(path, exist_ok=True)
        
        # Guardar cada modelo
        for product_id, model in self.product_models.items():
            # Crear subdirectorio para este producto
            product_path = os.path.join(path, f"product_{product_id}")
            os.makedirs(product_path, exist_ok=True)
            
            # Guardar modelo
            model.save(os.path.join(product_path, "model.h5"))
            
            # Guardar parámetros de normalización
            if product_id in self.data_processor.product_scalers:
                scaler = self.data_processor.product_scalers[product_id]
                scaler_params = {
                    'mean': scaler['mean'].to_dict(),
                    'std': scaler['std'].to_dict()
                }
                
                with open(os.path.join(product_path, "scaler_params.json"), 'w') as f:
                    json.dump(scaler_params, f)
                    
        logging.info(f"Modelos de productos guardados en {path}")
    
    def load_model(self, path="models/forecast"):
        """
        Carga un modelo guardado previamente.
        
        Args:
            path: Ruta donde está guardado el modelo
        """
        # Cargar modelo
        model_path = os.path.join(path, "tf_forecaster.h5")
        self.model = load_model(model_path)
        
        # Cargar parámetros de normalización
        scaler_path = os.path.join(path, "scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
                
            self.data_processor.mean = pd.Series(scaler_params['mean'])
            self.data_processor.std = pd.Series(scaler_params['std'])
            
        logging.info(f"Modelo cargado desde {path}")
    
    def load_product_models(self, path="models/forecast/products"):
        """
        Carga modelos de productos guardados previamente.
        
        Args:
            path: Ruta base donde están guardados los modelos
        """
        # Verificar si el directorio existe
        if not os.path.exists(path):
            raise ValueError(f"El directorio {path} no existe.")
            
        # Obtener subdirectorios (un directorio por producto)
        product_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        for product_dir in product_dirs:
            try:
                # Extraer ID del producto del nombre del directorio
                product_id = int(product_dir.split("_")[1])
                
                # Cargar modelo
                model_path = os.path.join(path, product_dir, "model.h5")
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    self.product_models[product_id] = model
                    
                    # Cargar parámetros de normalización
                    scaler_path = os.path.join(path, product_dir, "scaler_params.json")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'r') as f:
                            scaler_params = json.load(f)
                            
                        self.data_processor.product_scalers[product_id] = {
                            'mean': pd.Series(scaler_params['mean']),
                            'std': pd.Series(scaler_params['std'])
                        }
                        
                    logging.info(f"Modelo de producto {product_id} cargado correctamente.")
                    
            except Exception as e:
                logging.error(f"Error al cargar modelo del directorio {product_dir}: {str(e)}")
                
        logging.info(f"Se cargaron {len(self.product_models)} modelos de productos.")
    
    def plot_forecast(self, history_days=30):
        """
        Genera visualización de predicciones.
        
        Args:
            history_days: Días de historia a mostrar
            
        Returns:
            matplotlib.figure.Figure: Figura con el gráfico
        """
        # Obtener datos históricos
        data = self.data_processor.fetch_historical_data()
        
        # Obtener predicciones
        predictions = self.predict_next_days()
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extraer últimos días de historia
        if len(data) > history_days:
            historical = data.iloc[-history_days:]
        else:
            historical = data
            
        # Graficar datos históricos
        ax.plot(historical['fecha'], historical['total'], 
                label='Datos históricos', color='blue', marker='o')
        
        # Preparar datos de predicciones
        future_dates = [datetime.strptime(p['fecha'], '%Y-%m-%d') for p in predictions]
        future_values = [p['prediccion'] for p in predictions]
        
        # Graficar predicciones
        ax.plot(future_dates, future_values, 
                label='Predicciones', color='red', linestyle='--', marker='x')
        
        # Añadir banda de confianza (simplificada)
        confidence = 0.15  # 15% de error
        upper_bound = [val * (1 + confidence) for val in future_values]
        lower_bound = [max(0, val * (1 - confidence)) for val in future_values]
        
        ax.fill_between(future_dates, lower_bound, upper_bound, 
                       color='red', alpha=0.2, label='Intervalo de confianza')
        
        # Configurar gráfico
        ax.set_title('Predicción de Ventas', fontsize=15)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Ventas totales', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Rotar etiquetas del eje x
        plt.xticks(rotation=45)
        
        # Ajustar diseño
        plt.tight_layout()
        
        return fig
    
    def plot_product_forecast(self, product_id, history_days=30):
        """
        Genera visualización de predicciones para un producto específico.
        
        Args:
            product_id: ID del producto
            history_days: Días de historia a mostrar
            
        Returns:
            matplotlib.figure.Figure: Figura con el gráfico
        """
        # Verificar si hay modelo para este producto
        if product_id not in self.product_models:
            raise ValueError(f"No hay modelo entrenado para el producto {product_id}")
            
        # Obtener datos históricos para este producto
        product_data = self.data_processor.fetch_product_historical_data(product_id=product_id)
        
        # Obtener predicciones
        predictions = self.predict_product_demand(product_id=product_id)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extraer últimos días de historia
        if len(product_data) > history_days:
            historical = product_data.iloc[-history_days:]
        else:
            historical = product_data
            
        # Obtener nombre del producto
        product_info = self.data_processor.get_product_info(product_id)
        product_name = product_info.get('nombre', f'Producto {product_id}')
        
        # Graficar datos históricos
        ax.plot(historical['fecha'], historical['total'], 
                label='Datos históricos', color='blue', marker='o')
        
        # Preparar datos de predicciones
        if not predictions.empty:
            future_dates = predictions['fecha'].tolist()
            future_values = predictions['prediccion'].tolist()
            
            # Graficar predicciones
            ax.plot(future_dates, future_values, 
                    label='Predicciones', color='red', linestyle='--', marker='x')
            
            # Añadir banda de confianza (simplificada)
            confidence = 0.15  # 15% de error
            upper_bound = [val * (1 + confidence) for val in future_values]
            lower_bound = [max(0, val * (1 - confidence)) for val in future_values]
            
            ax.fill_between(future_dates, lower_bound, upper_bound, 
                           color='red', alpha=0.2, label='Intervalo de confianza')
        
        # Configurar gráfico
        ax.set_title(f'Predicción de Ventas - {product_name}', fontsize=15)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Ventas totales', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Rotar etiquetas del eje x
        plt.xticks(rotation=45)
        
        # Ajustar diseño
        plt.tight_layout()
        
        return fig
    
    def plot_category_forecast(self, category_id, history_days=30):
        """
        Genera visualización de predicciones para una categoría específica.
        
        Args:
            category_id: ID de la categoría
            history_days: Días de historia a mostrar
            
        Returns:
            matplotlib.figure.Figure: Figura con el gráfico
        """
        # Obtener predicciones por categoría
        category_predictions = self.predict_category_demand(category_id=category_id)
        
        if not category_predictions:
            raise ValueError(f"No hay predicciones disponibles para la categoría {category_id}")
            
        # Obtener información de la categoría
        category_info = self.data_processor.get_category_info(category_id)
        category_name = category_info.get('nombre', f'Categoría {category_id}')
        
        # Obtener datos históricos por producto y filtrar por esta categoría
        product_data = self.data_processor.fetch_product_historical_data()
        
        if 'categoria_id' in product_data.columns:
            # Filtrar por categoría
            category_data = product_data[product_data['categoria_id'] == category_id]
            
            # Agrupar por fecha para obtener el total diario de la categoría
            if not category_data.empty:
                historical_category = category_data.groupby('fecha')['total'].sum().reset_index()
            else:
                historical_category = pd.DataFrame(columns=['fecha', 'total'])
        else:
            # Si no tenemos categoría en los datos, crear dataframe vacío
            historical_category = pd.DataFrame(columns=['fecha', 'total'])
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extraer últimos días de historia si hay datos
        if not historical_category.empty:
            if len(historical_category) > history_days:
                historical = historical_category.iloc[-history_days:]
            else:
                historical = historical_category
                
            # Graficar datos históricos
            ax.plot(historical['fecha'], historical['total'], 
                    label='Datos históricos', color='blue', marker='o')
        
        # Preparar datos de predicciones
        future_dates = [datetime.strptime(p['fecha'], '%Y-%m-%d') if isinstance(p['fecha'], str) else p['fecha'] 
                        for p in category_predictions]
        future_values = [p['prediccion'] for p in category_predictions]
        
        # Graficar predicciones
        ax.plot(future_dates, future_values, 
                label='Predicciones', color='red', linestyle='--', marker='x')
        
        # Añadir banda de confianza (simplificada)
        confidence = 0.20  # 20% de error
        upper_bound = [val * (1 + confidence) for val in future_values]
        lower_bound = [max(0, val * (1 - confidence)) for val in future_values]
        
        ax.fill_between(future_dates, lower_bound, upper_bound, 
                       color='red', alpha=0.2, label='Intervalo de confianza')
        
        # Configurar gráfico
        ax.set_title(f'Predicción de Ventas - {category_name}', fontsize=15)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Ventas totales', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Rotar etiquetas del eje x
        plt.xticks(rotation=45)
        
        # Ajustar diseño
        plt.tight_layout()
        
        return fig
    
    def save_predictions_to_db(self, predictions, collection_name="predicciones_ventas"):
        """
        Guarda las predicciones en MongoDB.
        
        Args:
            predictions: Lista de predicciones
            collection_name: Nombre de la colección donde guardar
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        # Preparar datos para inserción
        documents = []
        
        for pred in predictions:
            doc = {
                "fecha": datetime.strptime(pred['fecha'], '%Y-%m-%d'),
                "dia": pred['dia'],
                "prediccion": pred['prediccion'],
                "confianza": pred['confianza'],
                "timestamp": datetime.now(),
                "tipo": "general"
            }
            documents.append(doc)
            
        # Eliminar predicciones anteriores del mismo tipo
        self.db_client.db[collection_name].delete_many({"tipo": "general"})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            logging.info(f"Guardadas {len(documents)} predicciones generales en MongoDB")
    
    def save_aggregated_predictions_to_db(self, predictions, period, collection_name="predicciones_ventas"):
        """
        Guarda las predicciones agregadas (semanales/mensuales) en MongoDB.
        
        Args:
            predictions: Lista de predicciones agregadas
            period: 'week' o 'month'
            collection_name: Nombre de la colección donde guardar
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        if not predictions:
            logging.warning(f"No hay predicciones {period} para guardar.")
            return
            
        # Determinar tipo según período
        tipo = "semanal" if period == "week" else "mensual"
        
        # Preparar datos para inserción
        documents = []
        
        for pred in predictions:
            doc = {
                "periodo": pred['periodo'],
                "fecha_inicio": datetime.strptime(pred['fecha_inicio'], '%Y-%m-%d'),
                "fecha_fin": datetime.strptime(pred['fecha_fin'], '%Y-%m-%d'),
                "prediccion": pred['prediccion'],
                "confianza": pred['confianza'],
                "timestamp": datetime.now(),
                "tipo": tipo
            }
            documents.append(doc)
            
        # Eliminar predicciones anteriores del mismo tipo
        self.db_client.db[collection_name].delete_many({"tipo": tipo})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            logging.info(f"Guardadas {len(documents)} predicciones {tipo}s en MongoDB")
            
    def save_product_predictions_to_db(self, predictions, collection_name="predicciones_productos"):
        """
        Guarda las predicciones por producto en MongoDB.
        
        Args:
            predictions: DataFrame con predicciones por producto
            collection_name: Nombre de la colección donde guardar
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        if predictions.empty:
            logging.warning("No hay predicciones por producto para guardar.")
            return
            
        # Preparar datos para inserción
        documents = []
        
        for _, row in predictions.iterrows():
            doc = {
                "fecha": row['fecha'] if isinstance(row['fecha'], datetime) else pd.to_datetime(row['fecha']),
                "dia": int(row['dia']),
                "producto_id": row['producto_id'],
                "nombre_producto": row['nombre_producto'],
                "prediccion": float(row['prediccion']),
                "confianza": float(row['confianza']),
                "timestamp": datetime.now()
            }
            
            # Agregar categoría si está disponible
            if 'categoria_id' in row and not pd.isna(row['categoria_id']):
                doc["categoria_id"] = row['categoria_id']
                
            documents.append(doc)
            
        # Eliminar predicciones anteriores
        self.db_client.db[collection_name].delete_many({})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            logging.info(f"Guardadas {len(documents)} predicciones por producto en MongoDB")
    
    def save_category_predictions_to_db(self, predictions, collection_name="predicciones_categoria"):
        """
        Guarda las predicciones por categoría en MongoDB.
        
        Args:
            predictions: Lista con predicciones por categoría
            collection_name: Nombre de la colección donde guardar
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        if not predictions:
            logging.warning("No hay predicciones por categoría para guardar.")
            return
            
        # Preparar datos para inserción
        documents = []
        
        for pred in predictions:
            doc = {
                "fecha": datetime.strptime(pred['fecha'], '%Y-%m-%d') if isinstance(pred['fecha'], str) else pred['fecha'],
                "dia": pred['dia'],
                "categoria_id": pred['categoria_id'],
                "nombre_categoria": pred['nombre_categoria'],
                "prediccion": float(pred['prediccion']),
                "confianza": float(pred['confianza']),
                "productos": int(pred['productos']),
                "timestamp": datetime.now()
            }
            documents.append(doc)
            
        # Eliminar predicciones anteriores
        self.db_client.db[collection_name].delete_many({})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            logging.info(f"Guardadas {len(documents)} predicciones por categoría en MongoDB")