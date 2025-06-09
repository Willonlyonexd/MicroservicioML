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
    Soporta arquitectura multi-tenant para separar datos por restaurante.
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
        self.logger = logging.getLogger(__name__)
        
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
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()]

        )
        
        return model
    
    def train(self, data=None, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size'], tenant_id=1):
        if data is None:
            self.logger.info(f"Obteniendo datos históricos para entrenar modelo general (tenant {tenant_id})")
            data = self.data_processor.fetch_historical_data(tenant_id=tenant_id)
        data_with_features = self.data_processor.add_time_features(data)
        X_train, y_train, X_val, y_val = self.data_processor.create_sequences(data_with_features)
        if len(X_train) == 0:
            raise ValueError(f"No hay suficientes datos para entrenar el modelo para tenant {tenant_id}. Se requieren al menos 14 días de datos históricos.")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=MODEL_PARAMS['patience'],
            restore_best_weights=True
        )
        callbacks = [early_stopping]
        if len(X_val) > 0:
            self.logger.info(f"Iniciando entrenamiento con validación para tenant {tenant_id}")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.logger.info(f"Iniciando entrenamiento sin validación para tenant {tenant_id}")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        self.logger.info(f"Entrenamiento completado para tenant {tenant_id}")
        return history

    def train_product_models(self, product_ids=None, top_n=10, epochs=MODEL_PARAMS['epochs'], tenant_id=1):
        # CAMBIO: Si no se proporcionan IDs específicos, obtener TODOS los productos para este tenant
        if product_ids is None:
            self.logger.info(f"Obteniendo TODOS los productos para tenant {tenant_id}")
            productos_cursor = self.db_client.db.raw_producto.find({"tenant_id": tenant_id}, {"producto_id": 1})
            product_ids = [p["producto_id"] for p in productos_cursor]
            self.logger.info(f"Se encontraron {len(product_ids)} productos para tenant {tenant_id}")
        if not product_ids:
            self.logger.warning(f"No se pudieron identificar productos para entrenamiento específico (tenant {tenant_id})")
            return {}
        self.logger.info(f"Obteniendo datos históricos por producto para tenant {tenant_id}")
        product_data = self.data_processor.fetch_product_historical_data(tenant_id=tenant_id)
        training_results = {}
        for product_id in product_ids:
            try:
                self.logger.info(f"Entrenando modelo para producto {product_id} (tenant {tenant_id})...")
                product_df = product_data[product_data['producto_id'] == product_id].copy()
                if len(product_df) < DATA_PARAMS['sequence_length'] + DATA_PARAMS['horizon'] + 5:
                    self.logger.warning(f"Datos insuficientes para producto {product_id} (tenant {tenant_id}). Omitiendo.")
                    training_results[product_id] = {
                        'status': 'skipped',
                        'reason': 'insufficient_data',
                        'tenant_id': tenant_id
                    }
                    continue
                product_df_features = self.data_processor.add_time_features(product_df)
                X_train, y_train, X_val, y_val = self.data_processor.create_product_sequences(
                    product_df_features, product_id)
                if X_train is None:
                    self.logger.warning(f"No se pudieron crear secuencias para producto {product_id} (tenant {tenant_id}). Omitiendo.")
                    training_results[product_id] = {
                        'status': 'skipped',
                        'reason': 'sequence_creation_failed',
                        'tenant_id': tenant_id
                    }
                    continue
                product_model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=MODEL_PARAMS['patience'],
                    restore_best_weights=True
                )
                if len(X_val) > 0:
                    history = product_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=min(32, len(X_train)),
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                else:
                    history = product_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=min(32, len(X_train)),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                self.product_models[product_id] = product_model
                training_results[product_id] = {
                    'status': 'success',
                    'val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                    'epochs': len(history.history['loss']),
                    'tenant_id': tenant_id
                }
                self.logger.info(f"Modelo para producto {product_id} (tenant {tenant_id}) entrenado exitosamente.")
            except Exception as e:
                self.logger.error(f"Error al entrenar modelo para producto {product_id} (tenant {tenant_id}): {str(e)}")
                training_results[product_id] = {
                    'status': 'error',
                    'error': str(e),
                    'tenant_id': tenant_id
                }
        return training_results
    
    def predict_next_days(self, days=DATA_PARAMS['horizon'], tenant_id=1):
        """
        Predice ventas para los próximos días a partir de la fecha actual.
        
        Args:
            days: Número de días a predecir (debe ser <= horizon del modelo)
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            dict: Predicciones para los próximos días
        """
        # Verificar que el modelo esté entrenado
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones.")
            
        # Obtener datos históricos para este tenant
        self.logger.info(f"Obteniendo datos históricos para predicciones (tenant {tenant_id})")
        data = self.data_processor.fetch_historical_data(tenant_id=tenant_id)
        
        # Agregar características temporales
        data_with_features = self.data_processor.add_time_features(data)
        
        # Preparar última secuencia
        last_sequence = self.data_processor.prepare_last_sequence(data_with_features)
        
        # Hacer predicción
        self.logger.info(f"Generando predicción para los próximos {days} días (tenant {tenant_id})")
        scaled_prediction = self.model.predict(last_sequence)
        
        # Invertir normalización - CORREGIDO: Aseguramos que la transformación inversa se aplique correctamente
        prediction = self.data_processor.inverse_transform(scaled_prediction[0])
        
        # Asegurar que no predecimos más días de los solicitados
        prediction = prediction[:days]
        
        # CAMBIO: Usar la fecha actual como base para las predicciones futuras
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        future_dates = [current_date + timedelta(days=i) for i in range(1, len(prediction) + 1)]
        
        # Formato de salida
        result = []
        for i, (date, pred) in enumerate(zip(future_dates, prediction)):
            result.append({
                'fecha': date,  # Ya no convertimos a string para mantener el tipo datetime
                'dia': i + 1,
                'prediccion': float(max(0, pred)),  # Asegurar valores no negativos
                'confianza': 0.9 - (i * 0.05),  # La confianza disminuye con días más lejanos
                'tenant_id': tenant_id,
                'generado_en': datetime.now(),
                'generado_por': f"ML System v1.0 - {os.getenv('USER', 'muimui69')}"
            })
        
        self.logger.info(f"Predicción completada para tenant {tenant_id}: {len(result)} días")
        return result
    
    def get_historical_and_forecast_data(self, history_days=30, forecast_days=7, tenant_id=1):
        """
        Obtiene datos históricos y predicciones futuras combinados.
        
        Args:
            history_days: Número de días de historia a incluir
            forecast_days: Número de días a predecir
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            dict: Datos históricos y predicciones combinados
        """
        # Fecha actual como punto pivote
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Obtener datos históricos hasta la fecha actual
        start_date = current_date - timedelta(days=history_days)
        historical_data = self.data_processor.fetch_historical_data(
            start_date=start_date,
            end_date=current_date,
            tenant_id=tenant_id
        )
        
        # Obtener predicciones futuras desde la fecha actual
        future_predictions = self.predict_next_days(days=forecast_days, tenant_id=tenant_id)
        
        # Preparar resultado combinado
        result = {
            'historical': [],
            'forecast': future_predictions,
            'current_date': current_date
        }
        
        # Formatear datos históricos
        for _, row in historical_data.iterrows():
            result['historical'].append({
                'fecha': row['fecha'],
                'ventas': float(row['total']),
                'tipo': 'REAL',
                'tenant_id': tenant_id
            })
        
        return result
    
    def predict_aggregated(self, period='day', horizon=None, tenant_id=1):
        """
        Predice ventas agregadas por día, semana o mes.
        
        Args:
            period: 'day', 'week' o 'month'
            horizon: Número de períodos a predecir (si es None, usar configuración por defecto)
            tenant_id: ID del tenant para filtrar datos
        
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
        
        # Obtener predicciones diarias para este tenant
        self.logger.info(f"Generando predicciones agregadas por {period} para tenant {tenant_id}")
        daily_predictions = self.predict_next_days(days=days_needed, tenant_id=tenant_id)
        
        if period == 'day':
            return daily_predictions
        
        # Para agregaciones semanales o mensuales
        result = []
        
        if period == 'week':
            # Agrupar por semanas (comenzando desde la fecha actual)
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Agrupar predicciones por semanas
            week_data = {}
            current_week = 1
            days_in_week = 0
            week_total = 0
            week_start = None
            
            for pred in daily_predictions:
                date = pred['fecha']
                
                if week_start is None:
                    week_start = date
                
                week_total += pred['prediccion']
                days_in_week += 1
                
                if days_in_week == 7:
                    # Completamos una semana
                    result.append({
                        'periodo': f"Semana {current_week}",
                        'fecha_inicio': week_start,
                        'fecha_fin': date,
                        'prediccion': week_total,
                        'confianza': 0.85 - (current_week * 0.05),  # Reducir confianza con el tiempo
                        'timestamp': datetime.now(),
                        'tenant_id': tenant_id  # Añadir tenant_id
                    })
                    
                    current_week += 1
                    days_in_week = 0
                    week_total = 0
                    week_start = None
            
            # Añadir última semana parcial si existe
            if days_in_week > 0:
                result.append({
                    'periodo': f"Semana {current_week}",
                    'fecha_inicio': week_start,
                    'fecha_fin': daily_predictions[-1]['fecha'],
                    'prediccion': week_total,
                    'confianza': 0.85 - (current_week * 0.05),
                    'timestamp': datetime.now(),
                    'tenant_id': tenant_id  # Añadir tenant_id
                })
        
        elif period == 'month':
            # Agrupar por meses (comenzando desde la fecha actual)
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Agrupar por meses
            month_data = {}
            
            for pred in daily_predictions:
                date = pred['fecha']
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
                    'periodo': f"Mes {i + 1}",
                    'fecha_inicio': data['start_date'],
                    'fecha_fin': data['end_date'],
                    'prediccion': data['total'],
                    'confianza': 0.8 - (i * 0.1),  # Reducir confianza con el tiempo
                    'timestamp': datetime.now(),
                    'tenant_id': tenant_id  # Añadir tenant_id
                })
        
        self.logger.info(f"Generadas {len(result)} predicciones agregadas por {period} para tenant {tenant_id}")
        return result
    
    def get_historical_and_aggregated_forecast(self, period='week', history_periods=12, forecast_periods=3, tenant_id=1):
        """
        Obtiene datos históricos y predicciones agregadas (semanas/meses) combinados.
        
        Args:
            period: 'week' o 'month'
            history_periods: Número de períodos históricos a incluir
            forecast_periods: Número de períodos a predecir
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            dict: Datos históricos y predicciones combinados
        """
        # Fecha actual como punto pivote
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calcular fecha de inicio según el período
        if period == 'week':
            # Comenzar desde X semanas atrás
            start_date = current_date - timedelta(days=history_periods * 7)
            period_name = "Semana"
        else:  # month
            # Calcular inicio aproximado para los últimos X meses
            start_date = current_date.replace(day=1)
            for _ in range(history_periods):
                # Retroceder al mes anterior (aproximado)
                if start_date.month == 1:
                    start_date = start_date.replace(year=start_date.year-1, month=12)
                else:
                    start_date = start_date.replace(month=start_date.month-1)
            period_name = "Mes"
        
        # Obtener datos históricos hasta la fecha actual
        historical_data = self.data_processor.fetch_historical_data(
            start_date=start_date,
            end_date=current_date,
            tenant_id=tenant_id
        )
        
        # Agrupar datos históricos por período
        historical_periods = []
        
        if period == 'week':
            # Agrupar por semanas
            week_data = {}
            for _, row in historical_data.iterrows():
                # Determinar la semana del año
                year_week = row['fecha'].strftime('%Y-W%U')
                
                if year_week not in week_data:
                    week_data[year_week] = {
                        'total': 0,
                        'start_date': row['fecha'],
                        'end_date': row['fecha']
                    }
                
                week_data[year_week]['total'] += row['total']
                
                # Actualizar fecha de fin si es posterior
                if row['fecha'] > week_data[year_week]['end_date']:
                    week_data[year_week]['end_date'] = row['fecha']
            
            # Convertir a lista de períodos
            for i, (year_week, data) in enumerate(sorted(week_data.items())):
                week_num = year_week.split('-W')[1]
                historical_periods.append({
                    'periodo': f"{period_name} {week_num}",
                    'fecha_inicio': data['start_date'],
                    'fecha_fin': data['end_date'],
                    'ventas': data['total'],
                    'tipo': 'REAL',
                    'tenant_id': tenant_id
                })
        
        else:  # month
            # Agrupar por meses
            month_data = {}
            for _, row in historical_data.iterrows():
                # Determinar el mes
                year_month = row['fecha'].strftime('%Y-%m')
                
                if year_month not in month_data:
                    month_data[year_month] = {
                        'total': 0,
                        'start_date': row['fecha'],
                        'end_date': row['fecha']
                    }
                
                month_data[year_month]['total'] += row['total']
                
                # Actualizar fecha de fin si es posterior
                if row['fecha'] > month_data[year_month]['end_date']:
                    month_data[year_month]['end_date'] = row['fecha']
            
            # Convertir a lista de períodos
            for year_month, data in sorted(month_data.items()):
                month_name = datetime.strptime(year_month, '%Y-%m').strftime('%B %Y')
                historical_periods.append({
                    'periodo': month_name,
                    'fecha_inicio': data['start_date'],
                    'fecha_fin': data['end_date'],
                    'ventas': data['total'],
                    'tipo': 'REAL',
                    'tenant_id': tenant_id
                })
        
        # Obtener predicciones agregadas futuras
        future_predictions = self.predict_aggregated(
            period=period, 
            horizon=forecast_periods,
            tenant_id=tenant_id
        )
        
        # Preparar resultado combinado
        result = {
            'historical': historical_periods,
            'forecast': future_predictions,
            'current_date': current_date
        }
        
        return result
    
    def predict_product_demand(self, product_id=None, days=DATA_PARAMS['horizon'], tenant_id=1):
        """
        Predice demanda por producto para los próximos días a partir de la fecha actual.
        
        Args:
            product_id: ID del producto específico (opcional)
            days: Número de días a predecir
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            pd.DataFrame: Predicciones por producto para los próximos días
        """
        # Fecha actual como punto pivote
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Si no se especifica producto, predecir para todos los productos con modelo
        if product_id is None:
            if not self.product_models:
                self.logger.warning(f"No hay modelos de productos entrenados para tenant {tenant_id}.")
                return pd.DataFrame()
                
            all_predictions = []
            
            # Obtener datos históricos por producto para este tenant
            self.logger.info(f"Obteniendo datos históricos por producto para tenant {tenant_id}")
            product_data = self.data_processor.fetch_product_historical_data(tenant_id=tenant_id)
            
            # Agregar características temporales
            product_data_features = self.data_processor.add_time_features(product_data)
            
            # Predecir para cada producto
            for prod_id, prod_model in self.product_models.items():
                try:
                    # Preparar última secuencia para este producto
                    last_sequence = self.data_processor.prepare_last_sequence(
                        product_data_features, product_id=prod_id)
                    
                    if last_sequence is None:
                        self.logger.warning(f"No se pudo preparar secuencia para producto {prod_id} (tenant {tenant_id}). Omitiendo.")
                        continue
                    
                    # Hacer predicción
                    scaled_prediction = prod_model.predict(last_sequence)
                    
                    # CORREGIDO: Invertir normalización correctamente
                    prediction = self.data_processor.inverse_transform(
                        scaled_prediction[0], product_id=prod_id)
                    
                    # Asegurar que no predecimos más días de los solicitados
                    prediction = prediction[:days]
                    
                    # Crear fechas futuras desde la fecha actual
                    future_dates = [current_date + timedelta(days=i+1) for i in range(len(prediction))]
                    
                    # Obtener información del producto
                    product_info = self.data_processor.get_product_info(prod_id, tenant_id=tenant_id)
                    
                    # Agregar predicciones a la lista
                    for i, (date, pred) in enumerate(zip(future_dates, prediction)):
                        all_predictions.append({
                            'fecha': date,
                            'dia': i + 1,
                            'producto_id': prod_id,
                            'nombre_producto': product_info.get('nombre', f'Producto {prod_id}'),
                            'categoria_id': product_info.get('categoria_id', None),
                            'prediccion': float(max(0, pred)),  # Asegurar valores no negativos
                            'confianza': 0.9 - (i * 0.05),  # La confianza disminuye con días más lejanos
                            'tenant_id': tenant_id  # Añadir tenant_id
                        })
                except Exception as e:
                    self.logger.error(f"Error al predecir para producto {prod_id} (tenant {tenant_id}): {str(e)}")
            
            # Convertir a DataFrame
            if all_predictions:
                return pd.DataFrame(all_predictions)
            else:
                return pd.DataFrame()
        else:
            # Predecir para un producto específico
            if product_id not in self.product_models:
                self.logger.warning(f"No hay modelo entrenado para producto {product_id} (tenant {tenant_id}).")
                return pd.DataFrame()
                
            # Obtener datos históricos para este producto y tenant
            self.logger.info(f"Generando predicción para producto {product_id} (tenant {tenant_id})")
            product_data = self.data_processor.fetch_product_historical_data(product_id=product_id, tenant_id=tenant_id)
            
            # Agregar características temporales
            product_data_features = self.data_processor.add_time_features(product_data)
            
            # Preparar última secuencia
            last_sequence = self.data_processor.prepare_last_sequence(
                product_data_features, product_id=product_id)
            
            if last_sequence is None:
                self.logger.warning(f"No se pudo preparar secuencia para producto {product_id} (tenant {tenant_id}).")
                return pd.DataFrame()
            
            # Hacer predicción
            scaled_prediction = self.product_models[product_id].predict(last_sequence)
            
            # CORREGIDO: Invertir normalización correctamente
            prediction = self.data_processor.inverse_transform(
                scaled_prediction[0], product_id=product_id)
            
            # Asegurar que no predecimos más días de los solicitados
            prediction = prediction[:days]
            
            # Crear fechas futuras desde la fecha actual
            future_dates = [current_date + timedelta(days=i+1) for i in range(len(prediction))]
            
            # Obtener información del producto
            product_info = self.data_processor.get_product_info(product_id, tenant_id=tenant_id)
            
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
                    'confianza': 0.9 - (i * 0.05),  # La confianza disminuye con días más lejanos
                    'tenant_id': tenant_id  # Añadir tenant_id
                })
            
            return pd.DataFrame(results)
    
    def get_product_historical_and_forecast(self, product_id, history_days=30, forecast_days=7, tenant_id=1):
        """
        Obtiene datos históricos y predicciones futuras para un producto específico.
        
        Args:
            product_id: ID del producto
            history_days: Número de días de historia a incluir
            forecast_days: Número de días a predecir
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            dict: Datos históricos y predicciones del producto combinados
        """
        # Fecha actual como punto pivote
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Obtener datos históricos del producto hasta la fecha actual
        start_date = current_date - timedelta(days=history_days)
        historical_data = self.data_processor.fetch_product_historical_data(
            product_id=product_id,
            start_date=start_date,
            end_date=current_date,
            tenant_id=tenant_id
        )
        
        # Obtener predicciones futuras para el producto
        future_predictions = self.predict_product_demand(
            product_id=product_id,
            days=forecast_days,
            tenant_id=tenant_id
        )
        
        # Obtener información del producto
        product_info = self.data_processor.get_product_info(product_id, tenant_id=tenant_id)
        
        # Preparar resultado combinado
        result = {
            'producto_id': product_id,  # CORREGIDO: Cambiado de 'product_id' a 'producto_id' para mantener consistencia
            'nombre_producto': product_info.get('nombre', f'Producto {product_id}'),  # CORREGIDO: Cambiado de 'product_name' a 'nombre_producto'
            'categoria_id': product_info.get('categoria_id', None),  # CORREGIDO: Cambiado de 'category_id' a 'categoria_id'
            'historical': [],
            'forecast': future_predictions.to_dict('records') if not future_predictions.empty else [],
            'current_date': current_date,
            'tenant_id': tenant_id
        }
        
        # Formatear datos históricos
        for _, row in historical_data.iterrows():
            result['historical'].append({
                'fecha': row['fecha'],
                'ventas': float(row['total']),
                'tipo': 'REAL',
                'tenant_id': tenant_id
            })
        
        return result
    
    def predict_category_demand(self, category_id=None, days=DATA_PARAMS['horizon'], tenant_id=1):
        """
        Predice demanda por categoría para los próximos días desde la fecha actual.
        
        Args:
            category_id: ID de la categoría específica (opcional)
            days: Número de días a predecir
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            list: Predicciones por categoría para los próximos días
        """
        # Fecha actual como punto pivote
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Obtener predicciones por producto para este tenant
        self.logger.info(f"Generando predicciones por categoría para tenant {tenant_id}")
        product_predictions = self.predict_product_demand(days=days, tenant_id=tenant_id)
        
        if product_predictions.empty:
            self.logger.warning(f"No hay predicciones por producto disponibles para tenant {tenant_id}.")
            return []
        
        # Verificar si tenemos categoría_id en las predicciones
        if 'categoria_id' not in product_predictions.columns:
            self.logger.warning(f"Las predicciones de productos no contienen información de categoría para tenant {tenant_id}.")
            return []
        
        # Si se especificó una categoría, filtrar predicciones
        if category_id is not None:
            self.logger.info(f"Filtrando predicciones para categoría {category_id}")
            product_predictions = product_predictions[product_predictions['categoria_id'] == category_id]
            
            if product_predictions.empty:
                self.logger.warning(f"No hay productos de la categoría {category_id} con predicciones disponibles para tenant {tenant_id}.")
                return []
        
        # Agrupar predicciones por fecha y categoría
        result = []
        
        # Convertir DataFrame a diccionario para facilitar manipulación
        pred_dict = {}
        for _, row in product_predictions.iterrows():
            date = row['fecha']
            category = row['categoria_id']
            
            # Omitir productos sin categoría
            if pd.isna(category):
                continue
            
            key = (date, category)
            if key not in pred_dict:
                # Obtener información de la categoría
                category_info = self.data_processor.get_category_info(category, tenant_id=tenant_id)
                category_name = category_info.get('nombre', f'Categoría {category}')
                
                pred_dict[key] = {
                    'fecha': date,
                    'dia': row['dia'],
                    'categoria_id': category,
                    'nombre_categoria': category_name,
                    'prediccion': 0,
                    'productos': 0,
                    'confianza': 0,
                    'tenant_id': tenant_id  # Añadir tenant_id
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
        
        self.logger.info(f"Generadas predicciones para {len(set(r['categoria_id'] for r in result))} categorías en tenant {tenant_id}")
        return result
    
    def get_category_historical_and_forecast(self, category_id, history_days=30, forecast_days=7, tenant_id=1):
        """
        Obtiene datos históricos y predicciones futuras para una categoría específica.
        
        Args:
            category_id: ID de la categoría
            history_days: Número de días de historia a incluir
            forecast_days: Número de días a predecir
            tenant_id: ID del tenant para filtrar datos
            
        Returns:
            dict: Datos históricos y predicciones de la categoría combinados
        """
        # Fecha actual como punto pivote
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Obtener datos históricos por producto y filtrar por esta categoría
        start_date = current_date - timedelta(days=history_days)
        product_data = self.data_processor.fetch_product_historical_data(
            start_date=start_date,
            end_date=current_date,
            tenant_id=tenant_id
        )
        
        # Filtrar por categoría y agrupar por fecha
        historical_data = []
        if 'categoria_id' in product_data.columns:
            category_data = product_data[product_data['categoria_id'] == category_id]
            
            if not category_data.empty:
                # Agrupar por fecha
                grouped = category_data.groupby('fecha')['total'].sum().reset_index()
                
                for _, row in grouped.iterrows():
                    historical_data.append({
                        'fecha': row['fecha'],
                        'ventas': float(row['total']),
                        'tipo': 'REAL',
                        'tenant_id': tenant_id
                    })
        
        # Obtener predicciones futuras para la categoría
        future_predictions = self.predict_category_demand(
            category_id=category_id,
            days=forecast_days,
            tenant_id=tenant_id
        )
        
        # Obtener información de la categoría
        category_info = self.data_processor.get_category_info(category_id, tenant_id=tenant_id)
        category_name = category_info.get('nombre', f'Categoría {category_id}')
        
        # Preparar resultado combinado
        result = {
            'category_id': category_id,
            'category_name': category_name,
            'historical': historical_data,
            'forecast': future_predictions,
            'current_date': current_date,
            'tenant_id': tenant_id
        }
        
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
        model_path = os.path.join(path, "tf_forecaster.h5")
        self.model.save(model_path)
        
        # Guardar parámetros de normalización
        scaler_params = {
            'mean': self.data_processor.mean.to_dict(),
            'std': self.data_processor.std.to_dict()
        }
        
        scaler_path = os.path.join(path, "scaler_params.json")
        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f)
            
        self.logger.info(f"Modelo guardado en {model_path}")
    
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
        
        models_saved = 0
        # Guardar cada modelo
        for product_id, model in self.product_models.items():
            # Crear subdirectorio para este producto
            product_path = os.path.join(path, f"product_{product_id}")
            os.makedirs(product_path, exist_ok=True)
            
            # Guardar modelo
            model_path = os.path.join(product_path, "model.h5")
            model.save(model_path)
            
            # Guardar parámetros de normalización
            if product_id in self.data_processor.product_scalers:
                scaler = self.data_processor.product_scalers[product_id]
                scaler_params = {
                    'mean': scaler['mean'].to_dict(),
                    'std': scaler['std'].to_dict()
                }
                
                scaler_path = os.path.join(product_path, "scaler_params.json")
                with open(scaler_path, 'w') as f:
                    json.dump(scaler_params, f)
                
                models_saved += 1
                    
        self.logger.info(f"Guardados {models_saved} modelos de productos en {path}")
    
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
            
        self.logger.info(f"Modelo cargado desde {model_path}")
    
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
        
        models_loaded = 0
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
                        
                    models_loaded += 1
                    
            except Exception as e:
                self.logger.error(f"Error al cargar modelo del directorio {product_dir}: {str(e)}")
                
        self.logger.info(f"Se cargaron {models_loaded} modelos de productos.")
    
    # Nota: Las funciones plot_* existen pero no deben usarse por ahora
    # Se mantienen para uso futuro cuando se requiera visualización gráfica
    def plot_forecast(self, history_days=30, forecast_days=7, tenant_id=1, generate_plot=False):
        """
        Obtiene datos para visualización de predicciones (sin generar gráfico).
        
        Args:
            history_days: Días de historia a incluir
            forecast_days: Días de predicción a incluir
            tenant_id: ID del tenant para filtrar datos
            generate_plot: Si es True, genera el gráfico (no usar por ahora)
            
        Returns:
            dict: Datos para visualización
        """
        # Obtener datos combinados
        combined_data = self.get_historical_and_forecast_data(
            history_days=history_days, 
            forecast_days=forecast_days,
            tenant_id=tenant_id
        )
        
        if not generate_plot:  # Por defecto solo devolvemos los datos
            return combined_data
            
        # El código para generar gráficos está presente pero no se usa por ahora
        # [CÓDIGO DE VISUALIZACIÓN OMITIDO]
        return None
    
    def plot_product_forecast(self, product_id, history_days=30, forecast_days=7, tenant_id=1, generate_plot=False):
        """
        Obtiene datos para visualización de predicciones de producto (sin generar gráfico).
        
        Args:
            product_id: ID del producto
            history_days: Días de historia a incluir
            forecast_days: Días de predicción a incluir
            tenant_id: ID del tenant para filtrar datos
            generate_plot: Si es True, genera el gráfico (no usar por ahora)
            
        Returns:
            dict: Datos para visualización
        """
        # Verificar si hay modelo para este producto
        if product_id not in self.product_models:
            raise ValueError(f"No hay modelo entrenado para el producto {product_id}")
            
        # Obtener datos combinados para este producto
        combined_data = self.get_product_historical_and_forecast(
            product_id=product_id,
            history_days=history_days, 
            forecast_days=forecast_days,
            tenant_id=tenant_id
        )
        
        if not generate_plot:  # Por defecto solo devolvemos los datos
            return combined_data
            
        # El código para generar gráficos está presente pero no se usa por ahora
        # [CÓDIGO DE VISUALIZACIÓN OMITIDO]
        return None
    
    def plot_category_forecast(self, category_id, history_days=30, forecast_days=7, tenant_id=1, generate_plot=False):
        """
        Obtiene datos para visualización de predicciones de categoría (sin generar gráfico).
        
        Args:
            category_id: ID de la categoría
            history_days: Número de días de historia a incluir
            forecast_days: Número de días a predecir
            tenant_id: ID del tenant para filtrar datos
            generate_plot: Si es True, genera el gráfico (no usar por ahora)
            
        Returns:
            dict: Datos para visualización
        """
        # Obtener datos combinados para esta categoría
        combined_data = self.get_category_historical_and_forecast(
            category_id=category_id,
            history_days=history_days, 
            forecast_days=forecast_days,
            tenant_id=tenant_id
        )
        
        if not generate_plot:  # Por defecto solo devolvemos los datos
            return combined_data
            
        # El código para generar gráficos está presente pero no se usa por ahora
        # [CÓDIGO DE VISUALIZACIÓN OMITIDO]
        return None
    
    def save_predictions_to_db(self, predictions, collection_name="predicciones_ventas", tenant_id=1):
        """
        Guarda las predicciones en MongoDB.
        
        Args:
            predictions: Lista de predicciones
            collection_name: Nombre de la colección donde guardar
            tenant_id: ID del tenant para filtrar datos
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        # Preparar datos para inserción
        documents = []
        
        for pred in predictions:
            doc = {
                "fecha": pred['fecha'],  # Ya es datetime
                "dia": pred['dia'],
                "prediccion": pred['prediccion'],
                "confianza": pred['confianza'],
                "timestamp": datetime.now(),
                "tipo": "general",
                "generado_en": pred.get('generado_en', datetime.now()),
                "generado_por": pred.get('generado_por', f"ML System v1.0 - {os.getenv('USER', 'muimui69')}"),
                "tenant_id": tenant_id  # Añadir tenant_id a cada documento
            }
            documents.append(doc)
            
        # Eliminar predicciones anteriores del mismo tipo para este tenant
        self.db_client.db[collection_name].delete_many({"tipo": "general", "tenant_id": tenant_id})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            self.logger.info(f"Guardadas {len(documents)} predicciones generales en MongoDB para tenant {tenant_id}")
    
    def save_aggregated_predictions_to_db(self, predictions, period, collection_name="ml_predicciones", tenant_id=1):
        """
        Guarda las predicciones agregadas (semanales/mensuales) en MongoDB.
        
        Args:
            predictions: Lista de predicciones agregadas
            period: 'week' o 'month'
            collection_name: Nombre de la colección donde guardar
            tenant_id: ID del tenant para filtrar datos
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        if not predictions:
            self.logger.warning(f"No hay predicciones {period} para tenant {tenant_id}.")
            return
            
        # Determinar tipo según período
        tipo = "semanal" if period == "week" else "mensual"
        
        # Preparar datos para inserción
        documents = []
        
        for pred in predictions:
            doc = {
                "periodo": pred['periodo'],
                "fecha_inicio": pred['fecha_inicio'],  # Ya es datetime
                "fecha_fin": pred['fecha_fin'],  # Ya es datetime
                "prediccion": pred['prediccion'],
                "confianza": pred['confianza'],
                "timestamp": datetime.now(),
                "tipo": tipo,
                "tenant_id": tenant_id  # Añadir tenant_id a cada documento
            }
            documents.append(doc)
            
        # Eliminar predicciones anteriores del mismo tipo para este tenant
        self.db_client.db[collection_name].delete_many({"tipo": tipo, "tenant_id": tenant_id})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            self.logger.info(f"Guardadas {len(documents)} predicciones {tipo}s en MongoDB para tenant {tenant_id}")
            
    def save_product_predictions_to_db(self, predictions, collection_name="predicciones_productos", tenant_id=1):
        """
        Guarda las predicciones por producto en MongoDB.
        
        Args:
            predictions: DataFrame con predicciones por producto
            collection_name: Nombre de la colección donde guardar
            tenant_id: ID del tenant para filtrar datos
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        if predictions.empty:
            self.logger.warning(f"No hay predicciones por producto para tenant {tenant_id}.")
            return
            
        # Preparar datos para inserción
        documents = []
        
        for _, row in predictions.iterrows():
            doc = {
                "fecha": row['fecha'],  # Ya es datetime
                "dia": int(row['dia']),
                "producto_id": row['producto_id'],
                "nombre_producto": row['nombre_producto'],
                "prediccion": float(row['prediccion']),
                "confianza": float(row['confianza']),
                "timestamp": datetime.now(),
                "tenant_id": tenant_id  # Añadir tenant_id a cada documento
            }
            
            # Agregar categoría si está disponible
            if 'categoria_id' in row and not pd.isna(row['categoria_id']):
                doc["categoria_id"] = row['categoria_id']
                
            documents.append(doc)
            
        # Eliminar predicciones anteriores para este tenant
        self.db_client.db[collection_name].delete_many({"tenant_id": tenant_id})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            self.logger.info(f"Guardadas {len(documents)} predicciones por producto en MongoDB para tenant {tenant_id}")
    
    def save_category_predictions_to_db(self, predictions, collection_name="ml_predicciones_categoria", tenant_id=1):
        """
        Guarda las predicciones por categoría en MongoDB.
        
        Args:
            predictions: Lista con predicciones por categoría
            collection_name: Nombre de la colección donde guardar
            tenant_id: ID del tenant para filtrar datos
        """
        if not self.db_client:
            raise ValueError("No hay conexión a base de datos.")
            
        if not predictions:
            self.logger.warning(f"No hay predicciones por categoría para tenant {tenant_id}.")
            return
            
        # Preparar datos para inserción
        documents = []
        
        for pred in predictions:
            doc = {
                "fecha": pred['fecha'],  # Ya es datetime
                "dia": pred['dia'],
                "categoria_id": pred['categoria_id'],
                "nombre_categoria": pred['nombre_categoria'],
                "prediccion": float(pred['prediccion']),
                "confianza": float(pred['confianza']),
                "productos": int(pred['productos']),
                "timestamp": datetime.now(),
                "tenant_id": tenant_id  # Añadir tenant_id a cada documento
            }
            documents.append(doc)
            
        # Eliminar predicciones anteriores para este tenant
        self.db_client.db[collection_name].delete_many({"tenant_id": tenant_id})
        
        # Insertar nuevas predicciones
        if documents:
            self.db_client.db[collection_name].insert_many(documents)
            self.logger.info(f"Guardadas {len(documents)} predicciones por categoría en MongoDB para tenant {tenant_id}")