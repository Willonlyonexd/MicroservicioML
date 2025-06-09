# ml/forecast/data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import logging

from .config import DATA_PARAMS, MONGO_PARAMS, PRODUCT_PARAMS


class TimeSeriesDataProcessor:
    def __init__(self, db_client):
        """
        Procesador de datos de serie temporal para forecast.
        
        Args:
            db_client: Cliente de MongoDB
        """
        self.db_client = db_client
        self.product_scalers = {}  # Almacena escaladores por producto
        
    def fetch_historical_data(self, months=24):
        """
        Obtiene datos históricos de ventas agregadas desde MongoDB.
        
        Args:
            months (int): Cantidad de meses a extraer
            
        Returns:
            pd.DataFrame: DataFrame con datos de ventas diarias
        """
        # Calcular fecha de inicio (hace X meses)
        start_date = datetime.now() - timedelta(days=30*months)
        print(f"Buscando datos desde: {start_date}")
        
        # Intentar obtener datos de raw_venta primero
        try:
            print("Intentando obtener datos de raw_venta...")
            # Consulta directa a raw_venta
            ventas_query = {
                "tenant_id": 1,  # Asumiendo tenant_id=1
                "fecha": {"$gte": start_date}
            }
            
            ventas_data = list(self.db_client.db.raw_venta.find(ventas_query))
            print(f"Encontrados {len(ventas_data)} registros en raw_venta")
            
            if ventas_data:
                # Convertir a DataFrame
                df = pd.DataFrame(ventas_data)
                print(f"Columnas disponibles en raw_venta: {df.columns.tolist()}")
                
                # Convertir fechas - adaptarse a la estructura real
                if 'fecha' in df.columns:
                    df['fecha'] = pd.to_datetime(df['fecha'])
                else:
                    # Buscar columna de fecha alternativa
                    for col in ['fecha_venta', 'created_at', 'date']:
                        if col in df.columns:
                            df['fecha'] = pd.to_datetime(df[col])
                            break
                
                # Identificar columna de monto
                amount_col = 'total'
                for col in ['monto', 'importe', 'valor', 'amount']:
                    if col in df.columns:
                        amount_col = col
                        break
                
                if amount_col in df.columns:
                    # Agrupar por fecha y sumar ventas
                    daily_sales = df.groupby(pd.Grouper(key='fecha', freq='D')).agg({
                        amount_col: 'sum'
                    }).reset_index()
                    
                    # Renombrar columna si es necesario
                    if amount_col != 'total':
                        daily_sales = daily_sales.rename(columns={amount_col: 'total'})
                    
                    # Añadir columna cantidad si no existe
                    if 'cantidad' not in daily_sales.columns:
                        daily_sales['cantidad'] = daily_sales['total'] / 100  # Estimación
                    
                    print(f"Datos procesados exitosamente de raw_venta: {len(daily_sales)} días")
                    return self._ensure_complete_dates(daily_sales)
        except Exception as e:
            print(f"Error al procesar raw_venta: {str(e)}")
        
        # Si llegamos aquí, intentamos con raw_pedido
        try:
            print("Intentando obtener datos de raw_pedido...")
            # Consulta directa a raw_pedido
            pedidos_query = {
                "tenant_id": 1,
                "$or": [
                    {"fecha_hora": {"$gte": start_date}},
                    {"fecha": {"$gte": start_date}}
                ]
            }
            
            pedidos_data = list(self.db_client.db.raw_pedido.find(pedidos_query))
            print(f"Encontrados {len(pedidos_data)} registros en raw_pedido")
            
            if pedidos_data:
                # Convertir a DataFrame
                df = pd.DataFrame(pedidos_data)
                print(f"Columnas disponibles en raw_pedido: {df.columns.tolist()}")
                
                # Convertir fechas - adaptarse a la estructura real
                if 'fecha_hora' in df.columns:
                    df['fecha'] = pd.to_datetime(df['fecha_hora']).dt.date
                elif 'fecha' in df.columns:
                    df['fecha'] = pd.to_datetime(df['fecha'])
                
                # Asegurarse que fecha sea datetime para agrupar
                df['fecha'] = pd.to_datetime(df['fecha'])
                
                # Identificar columna de monto
                amount_col = 'total'
                for col in ['total', 'monto', 'importe', 'valor', 'amount']:
                    if col in df.columns:
                        amount_col = col
                        break
                
                if amount_col in df.columns:
                    # Agrupar por fecha y sumar ventas
                    daily_sales = df.groupby(pd.Grouper(key='fecha', freq='D')).agg({
                        amount_col: 'sum'
                    }).reset_index()
                    
                    # Renombrar columna si es necesario
                    if amount_col != 'total':
                        daily_sales = daily_sales.rename(columns={amount_col: 'total'})
                    
                    # Añadir columna cantidad si no existe
                    if 'cantidad' not in daily_sales.columns:
                        daily_sales['cantidad'] = daily_sales['total'] / 100  # Estimación
                    
                    print(f"Datos procesados exitosamente de raw_pedido: {len(daily_sales)} días")
                    return self._ensure_complete_dates(daily_sales)
        except Exception as e:
            print(f"Error al procesar raw_pedido: {str(e)}")
        
        # Si llegamos aquí, generamos datos sintéticos
        print("No se pudieron obtener datos reales. Generando datos sintéticos...")
        return self._generate_synthetic_data(start_date)
    
    def fetch_product_historical_data(self, product_id=None, top_n=None, months=24):
        """
        Obtiene datos históricos de ventas por producto desde MongoDB.
        
        Args:
            product_id (int, optional): ID del producto específico a obtener
            top_n (int, optional): Número de productos más vendidos a incluir
            months (int): Cantidad de meses a extraer
            
        Returns:
            pd.DataFrame: DataFrame con datos de ventas diarias por producto
        """
        # Calcular fecha de inicio
        start_date = datetime.now() - timedelta(days=30*months)
        print(f"Buscando datos de productos desde: {start_date}")
        
        # Primero intentamos obtener detalles de pedidos (raw_pedido_detalle)
        try:
            print("Intentando obtener datos de pedidos con detalles...")
            
            # Consulta para obtener pedidos
            pedidos_query = {
                "tenant_id": 1,
                "$or": [
                    {"fecha_hora": {"$gte": start_date}},
                    {"fecha": {"$gte": start_date}}
                ]
            }
            
            # Obtener todos los pedidos
            pedidos_data = list(self.db_client.db.raw_pedido.find(pedidos_query, 
                                                                  {"_id": 1, "fecha": 1, "fecha_hora": 1}))
            
            if not pedidos_data:
                print("No se encontraron pedidos para el período solicitado")
                return self._generate_synthetic_product_data(product_id, start_date)
                
            # Extraer IDs de pedidos
            pedido_ids = [p["_id"] for p in pedidos_data]
            
            # Crear mapeo de id_pedido -> fecha
            fecha_por_pedido = {}
            for p in pedidos_data:
                if "fecha" in p:
                    fecha_por_pedido[p["_id"]] = p["fecha"]
                elif "fecha_hora" in p:
                    fecha_por_pedido[p["_id"]] = p["fecha_hora"]
            
            # Consulta para obtener detalles
            detalles_query = {
                "pedido_id": {"$in": pedido_ids}
            }
            
            if product_id:
                detalles_query["producto_id"] = product_id
                
            # Obtener detalles de pedidos
            detalles_data = list(self.db_client.db.raw_pedido_detalle.find(detalles_query))
            
            if not detalles_data:
                print("No se encontraron detalles de pedidos para el período solicitado")
                return self._generate_synthetic_product_data(product_id, start_date)
            
            # Convertir a DataFrame
            detalles_df = pd.DataFrame(detalles_data)
            
            # Añadir fecha desde el mapeo
            detalles_df["fecha"] = detalles_df["pedido_id"].map(fecha_por_pedido)
            
            # Convertir a datetime
            detalles_df["fecha"] = pd.to_datetime(detalles_df["fecha"])
            
            # Identificar columnas
            qty_col = "cantidad"
            price_col = "precio"
            product_col = "producto_id"
            
            for col in detalles_df.columns:
                if col.lower() in ["cantidad", "qty", "quantity"]:
                    qty_col = col
                elif col.lower() in ["precio", "precio_unitario", "price", "unit_price"]:
                    price_col = col
                elif col.lower() in ["producto_id", "product_id"]:
                    product_col = col
            
            # Calcular total por línea si no existe
            if "total_linea" not in detalles_df.columns:
                detalles_df["total_linea"] = detalles_df[qty_col] * detalles_df[price_col]
            
            # Agrupar por fecha y producto
            daily_product_sales = detalles_df.groupby([pd.Grouper(key="fecha", freq="D"), product_col]).agg({
                qty_col: "sum",
                "total_linea": "sum"
            }).reset_index()
            
            # Renombrar columnas para consistencia
            daily_product_sales = daily_product_sales.rename(columns={
                qty_col: "cantidad",
                "total_linea": "total",
                product_col: "producto_id"
            })
            
            # Si se solicitó solo los top productos
            if top_n and not product_id:
                # Identificar top productos por ventas totales
                top_products = daily_product_sales.groupby("producto_id")["total"].sum().nlargest(top_n).index.tolist()
                daily_product_sales = daily_product_sales[daily_product_sales["producto_id"].isin(top_products)]
            
            # Obtener información adicional de productos
            try:
                productos_data = list(self.db_client.db.raw_producto.find(
                    {"tenant_id": 1} if not product_id else {"tenant_id": 1, "_id": product_id}
                ))
                
                if productos_data:
                    productos_df = pd.DataFrame(productos_data)
                    
                    # Identificar columnas de nombre y categoría
                    name_col = "nombre"
                    for col in productos_df.columns:
                        if col.lower() in ["nombre", "name", "descripcion", "description"]:
                            name_col = col
                    
                    # Crear mapeo id -> nombre
                    if "_id" in productos_df.columns and name_col in productos_df.columns:
                        id_to_name = dict(zip(productos_df["_id"], productos_df[name_col]))
                        daily_product_sales["nombre_producto"] = daily_product_sales["producto_id"].map(id_to_name)
            except Exception as e:
                print(f"Error al obtener información de productos: {str(e)}")
            
            # Asegurar que existan todas las combinaciones fecha-producto
            complete_product_data = self._ensure_complete_product_dates(daily_product_sales)
            
            print(f"Datos de productos procesados exitosamente: {len(complete_product_data)} registros")
            return complete_product_data
            
        except Exception as e:
            print(f"Error al procesar datos de productos: {str(e)}")
            return self._generate_synthetic_product_data(product_id, start_date)
    
    def _ensure_complete_dates(self, daily_sales):
        """
        Asegura que no haya días faltantes en el DataFrame.
        
        Args:
            daily_sales (pd.DataFrame): DataFrame con datos diarios
            
        Returns:
            pd.DataFrame: DataFrame con fechas completas
        """
        # Asegurar que no haya días faltantes
        date_range = pd.date_range(start=daily_sales['fecha'].min(), 
                                  end=daily_sales['fecha'].max(), 
                                  freq='D')
        
        daily_sales = daily_sales.set_index('fecha').reindex(date_range).fillna(0).reset_index()
        daily_sales = daily_sales.rename(columns={'index': 'fecha'})
        
        return daily_sales
    
    def _ensure_complete_product_dates(self, product_sales):
        """
        Asegura que no haya combinaciones fecha-producto faltantes.
        
        Args:
            product_sales (pd.DataFrame): DataFrame con datos por producto
            
        Returns:
            pd.DataFrame: DataFrame completo con todas las combinaciones
        """
        # Obtener lista única de productos
        products = product_sales['producto_id'].unique()
        
        # Crear rango de fechas completo
        date_range = pd.date_range(start=product_sales['fecha'].min(), 
                                 end=product_sales['fecha'].max(), 
                                 freq='D')
        
        # Crear todas las combinaciones producto-fecha
        product_date_combinations = pd.MultiIndex.from_product(
            [products, date_range], 
            names=['producto_id', 'fecha']
        )
        
        # Crear DataFrame con índice MultiIndex
        complete_df = pd.DataFrame(index=product_date_combinations).reset_index()
        
        # Fusionar con datos originales
        result = pd.merge(
            complete_df, 
            product_sales, 
            on=['producto_id', 'fecha'], 
            how='left'
        )
        
        # Llenar valores nulos
        result = result.fillna({
            'total': 0,
            'cantidad': 0
        })
        
        # Preservar nombre_producto si existe
        if 'nombre_producto' in product_sales.columns:
            # Crear mapeo producto_id -> nombre_producto
            product_names = product_sales.dropna(subset=['nombre_producto'])
            product_name_map = dict(zip(product_names['producto_id'], product_names['nombre_producto']))
            
            # Aplicar mapeo
            result['nombre_producto'] = result['producto_id'].map(product_name_map)
        
        return result
    
    def _generate_synthetic_data(self, start_date):
        """
        Genera datos sintéticos para desarrollo cuando no hay datos reales.
        
        Args:
            start_date (datetime): Fecha de inicio para los datos sintéticos
            
        Returns:
            pd.DataFrame: DataFrame con datos sintéticos
        """
        # Crear rango de fechas desde start_date hasta hoy
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Crear tendencia base
        n_days = len(date_range)
        trend = np.linspace(100, 300, n_days) + np.random.normal(0, 10, n_days)
        
        # Añadir estacionalidad semanal
        weekday_effect = np.array([0.8, 0.9, 1.0, 1.0, 1.2, 1.5, 1.3])
        weekday_pattern = np.tile(weekday_effect, (n_days // 7) + 1)[:n_days]
        
        # Combinar elementos
        sales = trend * weekday_pattern
        
        # Crear DataFrame
        df = pd.DataFrame({
            'fecha': date_range,
            'total': sales,
            'cantidad': sales / 100  # Cantidad aproximada
        })
        
        print(f"Datos sintéticos generados: {len(df)} días")
        return df
    
    def _generate_synthetic_product_data(self, product_id=None, start_date=None):
        """
        Genera datos sintéticos por producto para desarrollo.
        
        Args:
            product_id (int, optional): ID de producto específico
            start_date (datetime): Fecha de inicio
            
        Returns:
            pd.DataFrame: DataFrame con datos sintéticos por producto
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
            
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)
        
        # Si no se especificó producto, generar datos para 5 productos
        if product_id is None:
            products = [101, 102, 103, 104, 105]
        else:
            products = [product_id]
            
        all_data = []
        
        for prod in products:
            # Base diferente para cada producto
            base = 50 + (prod % 10) * 20
            
            # Tendencia con crecimiento ligero
            trend = np.linspace(base, base * 1.5, n_days) + np.random.normal(0, base * 0.05, n_days)
            
            # Estacionalidad semanal
            weekday_effect = np.array([0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 1.4])
            weekday_pattern = np.tile(weekday_effect, (n_days // 7) + 1)[:n_days]
            
            # Añadir estacionalidad mensual (mayor demanda a fin de mes)
            day_of_month = np.array([date.day for date in date_range])
            month_effect = 1 + 0.2 * (day_of_month > 25)
            
            # Combinar elementos
            sales = trend * weekday_pattern * month_effect
            
            # Añadir algo de ruido
            sales = sales + np.random.normal(0, base * 0.1, n_days)
            sales = np.maximum(sales, 0)  # No permitir ventas negativas
            
            # Crear DataFrame para este producto
            prod_df = pd.DataFrame({
                'fecha': date_range,
                'producto_id': prod,
                'nombre_producto': f'Producto Sintético {prod}',
                'total': sales,
                'cantidad': np.round(sales / (base * 0.1))  # Cantidad aproximada
            })
            
            all_data.append(prod_df)
        
        # Combinar todos los productos
        result = pd.concat(all_data, ignore_index=True)
        print(f"Datos sintéticos de productos generados: {len(result)} registros para {len(products)} productos")
        
        return result
    
    def add_time_features(self, df):
        """
        Añade características temporales al dataset.
        
        Args:
            df (pd.DataFrame): DataFrame con columna 'fecha'
            
        Returns:
            pd.DataFrame: DataFrame con características temporales añadidas
        """
        # Crear copia para no modificar el original
        df_features = df.copy()
        
        # Extraer características de fecha
        df_features['day_of_week'] = df_features['fecha'].dt.dayofweek
        df_features['month'] = df_features['fecha'].dt.month
        df_features['day'] = df_features['fecha'].dt.day
        df_features['is_weekend'] = df_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # TODO: Añadir días festivos desde una fuente externa o base de datos
        df_features['is_holiday'] = 0
        
        return df_features
    
    def create_sequences(self, data, target_col='total'):
        """
        Crea secuencias para entrenamiento de modelo LSTM.
        
        Args:
            data (pd.DataFrame): DataFrame con datos de ventas y características
            target_col (str): Columna objetivo a predecir
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val) - Datos procesados para TensorFlow
        """
        # Extraer parámetros
        seq_length = DATA_PARAMS['sequence_length']
        horizon = DATA_PARAMS['horizon']
        features = DATA_PARAMS['features'] + [target_col]
        
        # Normalizar datos (target y características numéricas)
        data_to_scale = data[features].copy()
        
        # Guardar scaler para uso posterior en predicción
        self.mean = data_to_scale.mean()
        self.std = data_to_scale.std()
        
        # Evitar división por cero
        self.std = self.std.replace(0, 1)
        
        # Normalizar
        data_scaled = (data_to_scale - self.mean) / self.std
        
        # Preparar secuencias
        X, y = [], []
        
        for i in range(len(data_scaled) - seq_length - horizon + 1):
            # Secuencia de entrada: todas las características 
            X.append(data_scaled.iloc[i:i+seq_length].values)
            
            # Secuencia objetivo: solo columna target para los próximos 'horizon' días
            y_seq = data_scaled.iloc[i+seq_length:i+seq_length+horizon][target_col].values
            y.append(y_seq)
            
        X = np.array(X)
        y = np.array(y)
        
        # Dividir en train y validation
        train_size = int(len(X) * DATA_PARAMS['train_split'])
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        return X_train, y_train, X_val, y_val
    
    def create_product_sequences(self, data, product_id, target_col='total'):
        """
        Crea secuencias para entrenamiento de modelo LSTM para un producto específico.
        
        Args:
            data (pd.DataFrame): DataFrame con datos de ventas por producto
            product_id (int): ID del producto a procesar
            target_col (str): Columna objetivo a predecir
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val) - Datos procesados para TensorFlow
        """
        # Extraer parámetros
        seq_length = DATA_PARAMS['sequence_length']
        horizon = DATA_PARAMS['horizon']
        features = DATA_PARAMS['features'] + [target_col]
        
        # Filtrar datos solo para este producto
        product_data = data[data['producto_id'] == product_id].sort_values('fecha')
        
        # Verificar si hay suficientes datos
        min_required = seq_length + horizon + 10  # +10 para tener algo de validación
        if len(product_data) < min_required:
            print(f"Advertencia: Datos insuficientes para producto {product_id}. Se necesitan al menos {min_required} registros, pero hay {len(product_data)}.")
            # Rellenar con datos sintéticos si es necesario
            if len(product_data) < seq_length + horizon:
                return None, None, None, None
        
        # Normalizar datos para este producto
        data_to_scale = product_data[features].copy()
        
        # Guardar scaler para este producto
        mean = data_to_scale.mean()
        std = data_to_scale.std()
        
        # Evitar división por cero
        std = std.replace(0, 1)
        
        # Guardar para uso posterior en predicción
        self.product_scalers[product_id] = {
            'mean': mean,
            'std': std
        }
        
        # Normalizar
        data_scaled = (data_to_scale - mean) / std
        
        # Preparar secuencias
        X, y = [], []
        
        for i in range(len(data_scaled) - seq_length - horizon + 1):
            # Secuencia de entrada: todas las características 
            X.append(data_scaled.iloc[i:i+seq_length].values)
            
            # Secuencia objetivo: solo columna target para los próximos 'horizon' días
            y_seq = data_scaled.iloc[i+seq_length:i+seq_length+horizon][target_col].values
            y.append(y_seq)
            
        X = np.array(X)
        y = np.array(y)
        
        # Dividir en train y validation
        train_size = int(len(X) * DATA_PARAMS['train_split'])
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        return X_train, y_train, X_val, y_val
    
    def inverse_transform(self, scaled_values, product_id=None):
        """
        Revierte la normalización para obtener valores reales.
        
        Args:
            scaled_values (np.array): Valores normalizados
            product_id (int, optional): ID del producto si es predicción por producto
            
        Returns:
            np.array: Valores en escala original
        """
        if product_id is not None and product_id in self.product_scalers:
            # Usar scaler específico del producto
            scaler = self.product_scalers[product_id]
            return scaled_values * scaler['std']['total'] + scaler['mean']['total']
        else:
            # Usar scaler general
            return scaled_values * self.std['total'] + self.mean['total']
    
    def prepare_last_sequence(self, data, target_col='total', product_id=None):
        """
        Prepara la última secuencia disponible para hacer predicciones futuras.
        
        Args:
            data (pd.DataFrame): DataFrame con datos históricos
            target_col (str): Columna objetivo
            product_id (int, optional): ID del producto si es predicción por producto
            
        Returns:
            np.array: Última secuencia para input del modelo
        """
        seq_length = DATA_PARAMS['sequence_length']
        features = DATA_PARAMS['features'] + [target_col]
        
        if product_id is not None:
            # Filtrar datos solo para este producto
            data = data[data['producto_id'] == product_id].sort_values('fecha')
            
            # Verificar si hay suficientes datos
            if len(data) < seq_length:
                print(f"Advertencia: Datos insuficientes para preparar secuencia del producto {product_id}")
                return None
            
            # Tomar últimos seq_length días
            last_data = data[features].iloc[-seq_length:].copy()
            
            # Normalizar con scaler específico del producto
            if product_id in self.product_scalers:
                scaler = self.product_scalers[product_id]
                last_data_scaled = (last_data - scaler['mean'][features]) / scaler['std'][features]
            else:
                # Si no hay scaler específico, usar el general
                last_data_scaled = (last_data - self.mean[features]) / self.std[features]
        else:
            # Tomar últimos seq_length días para predicción general
            last_data = data[features].iloc[-seq_length:].copy()
            
            # Normalizar
            last_data_scaled = (last_data - self.mean[features]) / self.std[features]
        
        # Convertir a formato numpy para el modelo
        return np.expand_dims(last_data_scaled.values, axis=0)
    
    def get_product_info(self, product_id):
        """
        Obtiene información adicional de un producto.
        
        Args:
            product_id (int): ID del producto
            
        Returns:
            dict: Información del producto (nombre, categoría, etc)
        """
        try:
            product_data = self.db_client.db[MONGO_PARAMS['collection_products']].find_one({"_id": product_id})
            
            if not product_data:
                return {"nombre": f"Producto {product_id}", "categoria": "Desconocida"}
            
            # Identificar campos relevantes
            info = {
                "nombre": None,
                "categoria": None,
                "precio": None
            }
            
            # Buscar nombre
            for field in ["nombre", "name", "descripcion", "description"]:
                if field in product_data:
                    info["nombre"] = product_data[field]
                    break
            
            # Buscar categoría
            for field in ["categoria_id", "category_id"]:
                if field in product_data:
                    info["categoria"] = product_data[field]
                    break
            
            # Buscar precio
            for field in ["precio", "price", "precio_venta"]:
                if field in product_data:
                    info["precio"] = product_data[field]
                    break
            
            return info
        except Exception as e:
            print(f"Error al obtener información del producto {product_id}: {str(e)}")
            return {"nombre": f"Producto {product_id}", "categoria": "Desconocida"}
    
    def get_top_products(self, limit=10):
    """
    Obtiene los productos más vendidos basado en datos históricos.
    
    Args:
        limit (int): Número de productos a retornar
            
    Returns:
        list: Lista de IDs de productos más vendidos
    """
    product_ids = []
    
    try:
        # ESTRATEGIA 1: Obtener productos desde raw_pedido_detalle (método principal)
        logging.info("Buscando top productos en raw_pedido_detalle...")
        
        # Pipeline de agregación más flexible (sin filtro de fecha para capturar más datos)
        pipeline = [
            {"$group": {
                "_id": "$producto_id",
                "total_vendido": {"$sum": "$total_linea"},
                "cantidad_total": {"$sum": "$cantidad"}
            }},
            {"$sort": {"total_vendido": -1}},
            {"$limit": limit}
        ]
        
        top_products = list(self.db_client.db.raw_pedido_detalle.aggregate(pipeline))
        
        if top_products:
            product_ids = [p["_id"] for p in top_products]
            logging.info(f"Encontrados {len(product_ids)} productos en raw_pedido_detalle")
            return product_ids
            
        # ESTRATEGIA 2: Si no hay detalles, buscar directamente en raw_producto
        logging.info("No se encontraron detalles de pedidos. Buscando en raw_producto...")
        
        # Obtener los primeros productos de la colección de productos
        productos = list(self.db_client.db.raw_producto.find({}).limit(limit))
        
        if productos:
            product_ids = [p["_id"] for p in productos]
            logging.info(f"Encontrados {len(product_ids)} productos en raw_producto")
            return product_ids
            
        # ESTRATEGIA 3: Buscar en otras colecciones (raw_inventario, ventas_diarias, etc.)
        for collection_name in ["raw_inventario", "ventas_diarias"]:
            if collection_name in self.db_client.db.list_collection_names():
                logging.info(f"Buscando productos en {collection_name}...")
                
                # Buscar campo de producto en la colección
                product_field = "producto_id"  # Nombre de campo por defecto
                
                # Obtener un documento de muestra para verificar estructura
                sample = self.db_client.db[collection_name].find_one({})
                if sample:
                    # Verificar campos disponibles
                    for field in sample.keys():
                        if "producto" in field.lower() or "product" in field.lower():
                            product_field = field
                            break
                
                # Agregación para encontrar productos únicos
                pipeline = [
                    {"$group": {"_id": f"${product_field}"}},
                    {"$limit": limit}
                ]
                
                unique_products = list(self.db_client.db[collection_name].aggregate(pipeline))
                
                if unique_products:
                    product_ids = [p["_id"] for p in unique_products if p["_id"] is not None]
                    if product_ids:
                        logging.info(f"Encontrados {len(product_ids)} productos en {collection_name}")
                        return product_ids
        
        # DIAGNÓSTICO: Listar todas las colecciones disponibles
        collections = self.db_client.db.list_collection_names()
        logging.warning(f"No se encontraron productos en ninguna colección. Colecciones disponibles: {collections}")
        
        # ÚLTIMO RECURSO: Crear IDs de producto artificiales, pero con advertencia clara
        logging.warning("ADVERTENCIA: No se encontraron productos reales. Usando IDs artificiales para demostración.")
        return [1001, 1002, 1003, 1004, 1005][:limit]
            
    except Exception as e:
        logging.error(f"Error al obtener productos más vendidos: {str(e)}", exc_info=True)
        logging.warning("ADVERTENCIA: Usando IDs artificiales debido a error en la consulta.")
        return [1001, 1002, 1003, 1004, 1005][:limit]