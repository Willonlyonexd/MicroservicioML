import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import logging
import random

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
        self.category_info = {}    # Almacena información de categorías
        self.logger = logging.getLogger(__name__)
        
    def fetch_historical_data(self, tenant_id=1, months=24, start_date=None, end_date=None):
        """
        Obtiene datos históricos de ventas agregadas desde MongoDB.
        
        Args:
            tenant_id (int): ID del tenant para filtrar datos
            months (int): Cantidad de meses a extraer
            start_date (datetime, optional): Fecha de inicio para filtrar datos
            end_date (datetime, optional): Fecha de fin para filtrar datos
            
        Returns:
            pd.DataFrame: DataFrame con datos de ventas diarias
        """
        # Usar fechas proporcionadas o calcular basado en la fecha actual
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            start_date = end_date - timedelta(days=30*months)
            
        self.logger.info(f"Buscando datos desde: {start_date} hasta: {end_date} para tenant {tenant_id}")
        
        # CAMBIADO: Intentar obtener datos de raw_pedido primero (invertir orden)
        try:
            self.logger.info(f"Intentando obtener datos de raw_pedido (fuente principal) para tenant {tenant_id}...")
            # Consulta directa a raw_pedido
            pedidos_query = {
                "tenant_id": tenant_id,  # Usar tenant_id como parámetro
                "$or": [
                    {"fecha_hora": {"$gte": start_date, "$lte": end_date}},
                    {"fecha": {"$gte": start_date, "$lte": end_date}}
                ]
            }
            
            pedidos_data = list(self.db_client.db.raw_pedido.find(pedidos_query))
            self.logger.info(f"Encontrados {len(pedidos_data)} registros en raw_pedido para tenant {tenant_id}")
            
            if pedidos_data:
                # Convertir a DataFrame
                df = pd.DataFrame(pedidos_data)
                self.logger.debug(f"Columnas disponibles en raw_pedido: {df.columns.tolist()}")
                
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
                    
                    # Añadir tenant_id para garantizar que se propague
                    daily_sales['tenant_id'] = tenant_id
                    
                    self.logger.info(f"Datos procesados exitosamente de raw_pedido: {len(daily_sales)} días")
                    days_count = len(daily_sales['fecha'].dt.date.unique())
                    self.logger.info(f"Días únicos encontrados: {days_count}")
                    
                    if days_count < 14:
                        self.logger.warning(f"Solo se encontraron {days_count} días únicos, se requieren al menos 14 para el modelo LSTM.")
                    
                    return self._ensure_complete_dates(daily_sales, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error al procesar raw_pedido para tenant {tenant_id}: {str(e)}")
        
        # Si llegamos aquí, intentamos con raw_venta
        try:
            self.logger.info(f"Intentando obtener datos de raw_venta (fuente secundaria) para tenant {tenant_id}...")
            # Consulta usando etl_timestamp en lugar de fecha
            ventas_query = {
                "tenant_id": tenant_id,  # Usar tenant_id como parámetro
                "etl_timestamp": {"$gte": start_date, "$lte": end_date}  # Usar etl_timestamp en lugar de fecha
            }
            
            ventas_data = list(self.db_client.db.raw_venta.find(ventas_query))
            self.logger.info(f"Encontrados {len(ventas_data)} registros en raw_venta para tenant {tenant_id}")
            
            if ventas_data:
                # Convertir a DataFrame
                df = pd.DataFrame(ventas_data)
                self.logger.debug(f"Columnas disponibles en raw_venta: {df.columns.tolist()}")
                
                # Usar etl_timestamp como fecha para análisis
                df['fecha'] = pd.to_datetime(df['etl_timestamp'])
                
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
                    
                    # Añadir tenant_id para garantizar que se propague
                    daily_sales['tenant_id'] = tenant_id
                    
                    days_count = len(daily_sales['fecha'].dt.date.unique())
                    self.logger.info(f"Datos procesados exitosamente de raw_venta: {len(daily_sales)} días, días únicos: {days_count}")
                    
                    if days_count < 14:
                        self.logger.warning(f"Solo se encontraron {days_count} días únicos, se requieren al menos 14 para el modelo LSTM.")
                        if days_count == 1:
                            self.logger.warning("Se detectó solo 1 día de datos. Se generarán datos sintéticos adicionales.")
                            # Generar datos sintéticos pero mantener el día real
                            real_day = daily_sales.iloc[0]
                            synthetic_data = self._generate_synthetic_data(start_date, end_date, tenant_id)
                            
                            # Mantener el día real y reemplazar un día sintético
                            synthetic_data.loc[synthetic_data['fecha'] >= real_day['fecha'], 'total'] = real_day['total']
                            return synthetic_data
                    
                    return self._ensure_complete_dates(daily_sales, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error al procesar raw_venta para tenant {tenant_id}: {str(e)}")
        
        # Si llegamos aquí, generamos datos sintéticos
        self.logger.warning(f"No se pudieron obtener datos reales suficientes para tenant {tenant_id}. Generando datos sintéticos...")
        return self._generate_synthetic_data(start_date, end_date, tenant_id)
    
    def fetch_product_historical_data(self, tenant_id=1, product_id=None, top_n=None, months=24, start_date=None, end_date=None):
        """
        Obtiene datos históricos de ventas por producto desde MongoDB.
        
        Args:
            tenant_id (int): ID del tenant para filtrar datos
            product_id (int, optional): ID del producto específico a obtener
            top_n (int, optional): Número de productos más vendidos a incluir
            months (int): Cantidad de meses a extraer
            start_date (datetime, optional): Fecha de inicio para filtrar datos
            end_date (datetime, optional): Fecha de fin para filtrar datos
            
        Returns:
            pd.DataFrame: DataFrame con datos de ventas diarias por producto
        """
        # Usar fechas proporcionadas o calcular basado en la fecha actual
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            start_date = end_date - timedelta(days=30*months)
            
        self.logger.info(f"Buscando datos de productos desde: {start_date} hasta: {end_date} para tenant {tenant_id}")
        
        # Primero intentamos obtener detalles de pedidos (raw_pedido_detalle)
        try:
            self.logger.info(f"Intentando obtener datos de pedidos con detalles para tenant {tenant_id}...")
            
            # Consulta para obtener pedidos
            pedidos_query = {
                "tenant_id": tenant_id,  # Usar tenant_id como parámetro
                "$or": [
                    {"fecha_hora": {"$gte": start_date, "$lte": end_date}},
                    {"fecha": {"$gte": start_date, "$lte": end_date}}
                ]
            }
            
            # CORREGIDO: Obtener todos los pedidos incluyendo el campo pedido_id
            pedidos_data = list(self.db_client.db.raw_pedido.find(pedidos_query, 
                                                                  {"_id": 1, "pedido_id": 1, "fecha": 1, "fecha_hora": 1}))
            
            if not pedidos_data:
                self.logger.warning(f"No se encontraron pedidos para el período solicitado para tenant {tenant_id}")
                return self._generate_synthetic_product_data(product_id, start_date, end_date, tenant_id)
                
            # CORREGIDO: Extraer IDs utilizando el campo pedido_id en lugar de _id
            # Los IDs en raw_pedido_detalle son enteros simples (pedido_id), no ObjectIDs
            pedido_ids = []
            for p in pedidos_data:
                if "pedido_id" in p:
                    pedido_ids.append(p["pedido_id"])
            
            if not pedido_ids:
                self.logger.warning("No se encontraron pedido_id válidos")
                return self._generate_synthetic_product_data(product_id, start_date, end_date, tenant_id)
            
            # CORREGIDO: Crear mapeo de pedido_id -> fecha (en lugar de _id -> fecha)
            fecha_por_pedido = {}
            for p in pedidos_data:
                if "pedido_id" not in p:
                    continue
                    
                if "fecha" in p:
                    fecha_por_pedido[p["pedido_id"]] = p["fecha"]
                elif "fecha_hora" in p:
                    fecha_por_pedido[p["pedido_id"]] = p["fecha_hora"]
            
            # Consulta para obtener detalles usando pedido_id
            detalles_query = {
                "pedido_id": {"$in": pedido_ids}
            }
            
            # Si se especificó un producto_id, añadirlo a la consulta
            if product_id:
                # Verificar si es un ObjectID o un entero
                if isinstance(product_id, str) and len(product_id) > 10:
                    # Es un ObjectID, buscar en productos para obtener el producto_id
                    producto = self.db_client.db.raw_producto.find_one({"_id": product_id})
                    if producto and "producto_id" in producto:
                        detalles_query["producto_id"] = producto["producto_id"]
                else:
                    # Es un entero o string simple
                    detalles_query["producto_id"] = int(product_id) if isinstance(product_id, str) and product_id.isdigit() else product_id
            
            # Obtener detalles de pedidos
            detalles_data = list(self.db_client.db.raw_pedido_detalle.find(detalles_query))
            
            if not detalles_data:
                self.logger.warning(f"No se encontraron detalles de pedidos para el período solicitado para tenant {tenant_id}. Query: {detalles_query}")
                # Intentar diagnóstico adicional
                sample_detalle = self.db_client.db.raw_pedido_detalle.find_one()
                if sample_detalle:
                    self.logger.debug(f"Ejemplo de documento en raw_pedido_detalle: {sample_detalle}")
                    if "pedido_id" in sample_detalle:
                        self.logger.debug(f"Tipo de pedido_id en detalles: {type(sample_detalle['pedido_id']).__name__}")
                
                return self._generate_synthetic_product_data(product_id, start_date, end_date, tenant_id)
            
            self.logger.info(f"Encontrados {len(detalles_data)} detalles de pedidos para tenant {tenant_id}")
            
            # Convertir a DataFrame
            detalles_df = pd.DataFrame(detalles_data)
            
            # Añadir fecha desde el mapeo
            detalles_df["fecha"] = detalles_df["pedido_id"].map(fecha_por_pedido)
            
            # Convertir a datetime
            detalles_df["fecha"] = pd.to_datetime(detalles_df["fecha"])
            
            # Identificar columnas
            qty_col = "cantidad"
            price_col = "precio_unitario"  # Cambiado a precio_unitario según ejemplo
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
                if price_col in detalles_df.columns and qty_col in detalles_df.columns:
                    detalles_df["total_linea"] = detalles_df[qty_col] * detalles_df[price_col]
                else:
                    # Si faltan columnas necesarias
                    if "subtotal" in detalles_df.columns:
                        detalles_df["total_linea"] = detalles_df["subtotal"]
                    else:
                        self.logger.warning(f"No se pudo calcular total_linea, faltan columnas. Columnas disponibles: {detalles_df.columns.tolist()}")
                        detalles_df["total_linea"] = 1.0  # Valor por defecto
            
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
            
            # Añadir tenant_id para garantizar que se propague
            daily_product_sales['tenant_id'] = tenant_id
            
            # Obtener información adicional de productos
            try:
                # CORREGIDO: Ajustar la consulta para buscar por producto_id no por _id
                productos_query = {"tenant_id": tenant_id}  # Añadir tenant_id
                if product_id:
                    # Si product_id es un entero simple o un string que se puede convertir a entero
                    if isinstance(product_id, int) or (isinstance(product_id, str) and product_id.isdigit()):
                        productos_query["producto_id"] = int(product_id) if isinstance(product_id, str) else product_id
                    else:
                        # Si es un ObjectID
                        productos_query["_id"] = product_id
                
                productos_data = list(self.db_client.db.raw_producto.find(productos_query))
                
                if productos_data:
                    productos_df = pd.DataFrame(productos_data)
                    self.logger.info(f"Información de productos obtenida: {len(productos_data)} productos para tenant {tenant_id}")
                    
                    # Identificar columnas de nombre, categoría
                    name_col = "nombre"
                    cat_col = "categoria_id"
                    id_col = "producto_id"  # Cambio importante: usar producto_id, no _id
                    
                    for col in productos_df.columns:
                        if col.lower() in ["nombre", "name", "descripcion", "description"]:
                            name_col = col
                        if col.lower() in ["categoria_id", "category_id", "categoria"]:
                            cat_col = col
                        if col.lower() in ["producto_id", "product_id"]:
                            id_col = col
                    
                    # CORREGIDO: Crear mapeo producto_id -> nombre, producto_id -> categoría
                    if id_col in productos_df.columns:
                        if name_col in productos_df.columns:
                            id_to_name = dict(zip(productos_df[id_col], productos_df[name_col]))
                            daily_product_sales["nombre_producto"] = daily_product_sales["producto_id"].map(id_to_name)
                        
                        if cat_col in productos_df.columns:
                            id_to_cat = dict(zip(productos_df[id_col], productos_df[cat_col]))
                            daily_product_sales["categoria_id"] = daily_product_sales["producto_id"].map(id_to_cat)
                            
                            # Almacenar mappings para categorías
                            self.category_info = self._get_category_info(productos_df, cat_col)
            except Exception as e:
                self.logger.error(f"Error al obtener información de productos para tenant {tenant_id}: {str(e)}")
            
            # Asegurar que existan todas las combinaciones fecha-producto
            complete_product_data = self._ensure_complete_product_dates(daily_product_sales, start_date, end_date)
            
            self.logger.info(f"Datos de productos procesados exitosamente: {len(complete_product_data)} registros para tenant {tenant_id}")
            return complete_product_data
            
        except Exception as e:
            self.logger.error(f"Error al procesar datos de productos para tenant {tenant_id}: {str(e)}")
            return self._generate_synthetic_product_data(product_id, start_date, end_date, tenant_id)
    
    def diagnosticar_detalles_pedidos(self, tenant_id=1):
        """
        Diagnostica problemas de relación entre pedidos y detalles.
        
        Args:
            tenant_id (int): ID del tenant para diagnosticar
        """
        try:
            # Obtener un pedido de muestra
            pedido = self.db_client.db.raw_pedido.find_one({"tenant_id": tenant_id})  # Añadir tenant_id
            if not pedido:
                self.logger.warning(f"No se encontraron pedidos para tenant {tenant_id}")
                return
                
            self.logger.info(f"Pedido ejemplo: _id={pedido.get('_id')}, pedido_id={pedido.get('pedido_id')}")
            
            # Buscar detalles con _id
            detalles_id = list(self.db_client.db.raw_pedido_detalle.find({"pedido_id": pedido.get('_id')}))
            self.logger.info(f"Detalles encontrados buscando por _id: {len(detalles_id)}")
            
            # Buscar detalles con pedido_id
            if "pedido_id" in pedido:
                detalles_pid = list(self.db_client.db.raw_pedido_detalle.find({"pedido_id": pedido.get('pedido_id')}))
                self.logger.info(f"Detalles encontrados buscando por pedido_id: {len(detalles_pid)}")
                
                if detalles_pid:
                    detalle = detalles_pid[0]
                    self.logger.debug(f"Ejemplo detalle: {detalle}")
                    
                    # Verificar producto
                    if "producto_id" in detalle:
                        producto_id = detalle.get("producto_id")
                        producto = self.db_client.db.raw_producto.find_one({"producto_id": producto_id, "tenant_id": tenant_id})  # Añadir tenant_id
                        if producto:
                            self.logger.info(f"Producto encontrado por producto_id: {producto.get('nombre', 'Sin nombre')}")
                        else:
                            self.logger.warning(f"No se encontró producto con producto_id={producto_id} para tenant {tenant_id}")
                            
                            # Intentar buscar por _id
                            producto = self.db_client.db.raw_producto.find_one({"_id": producto_id, "tenant_id": tenant_id})  # Añadir tenant_id
                            if producto:
                                self.logger.info(f"Producto encontrado por _id: {producto.get('nombre', 'Sin nombre')}")
        except Exception as e:
            self.logger.error(f"Error en diagnóstico para tenant {tenant_id}: {str(e)}")
    
    def _get_category_info(self, productos_df, cat_col):
        """
        Extrae información de categorías para uso futuro.
        
        Args:
            productos_df: DataFrame con datos de productos
            cat_col: Nombre de la columna de categoría
            
        Returns:
            dict: Información de categorías
        """
        category_info = {}
        
        # Si no hay columna de categoría, devolver vacío
        if cat_col not in productos_df.columns:
            return category_info
            
        # Crear información por categoría
        for cat_id in productos_df[cat_col].unique():
            if pd.isna(cat_id):
                continue
                
            # Filtrar productos de esta categoría
            cat_products = productos_df[productos_df[cat_col] == cat_id]
            
            # Buscar nombre de categoría si existe
            cat_name = f"Categoría {cat_id}"
            for col in productos_df.columns:
                if "categoria" in col.lower() and "nombre" in col.lower():
                    if col in cat_products.columns and not cat_products[col].isnull().all():
                        cat_name = cat_products[col].iloc[0]
                        break
            
            # Identificar la columna de ID del producto
            id_col = "producto_id" if "producto_id" in productos_df.columns else "_id"
            
            # Guardar información
            category_info[cat_id] = {
                "nombre": cat_name,
                "productos": cat_products[id_col].tolist()
            }
            
        return category_info
    
    def _ensure_complete_dates(self, daily_sales, start_date=None, end_date=None):
        """
        Asegura que no haya días faltantes en el DataFrame.
        
        Args:
            daily_sales (pd.DataFrame): DataFrame con datos diarios
            start_date (datetime, optional): Fecha de inicio
            end_date (datetime, optional): Fecha de fin
            
        Returns:
            pd.DataFrame: DataFrame con fechas completas
        """
        # Preservar tenant_id
        tenant_id = daily_sales['tenant_id'].iloc[0] if 'tenant_id' in daily_sales.columns else 1
        
        # Determinar rango de fechas
        if start_date is None:
            start_date = daily_sales['fecha'].min()
            
        if end_date is None:
            end_date = daily_sales['fecha'].max()
            
        # Asegurar que start_date y end_date sean datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Crear rango de fechas completo
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Asegurar que daily_sales tiene fecha como datetime
        daily_sales['fecha'] = pd.to_datetime(daily_sales['fecha'])
        
        # Reindexar para incluir todas las fechas
        daily_sales = daily_sales.set_index('fecha').reindex(date_range).reset_index()
        daily_sales = daily_sales.rename(columns={'index': 'fecha'})
        
        # Llenar valores nulos
        for col in daily_sales.columns:
            if col != 'fecha':
                if col in ['total', 'cantidad']:
                    daily_sales[col] = daily_sales[col].fillna(0)
                else:
                    # Otros campos (como tenant_id)
                    if col == 'tenant_id':
                        daily_sales[col] = daily_sales[col].fillna(tenant_id)
                    else:
                        daily_sales[col] = daily_sales[col].fillna(daily_sales[col].iloc[0] if len(daily_sales) > 0 else None)
        
        # Asegurar que tenant_id exista y tenga el valor correcto
        if 'tenant_id' not in daily_sales.columns:
            daily_sales['tenant_id'] = tenant_id
        
        return daily_sales
    
    def _ensure_complete_product_dates(self, product_sales, start_date=None, end_date=None):
        """
        Asegura que no haya combinaciones fecha-producto faltantes.
        
        Args:
            product_sales (pd.DataFrame): DataFrame con datos por producto
            start_date (datetime, optional): Fecha de inicio
            end_date (datetime, optional): Fecha de fin
            
        Returns:
            pd.DataFrame: DataFrame completo con todas las combinaciones
        """
        # Preservar tenant_id
        tenant_id = product_sales['tenant_id'].iloc[0] if 'tenant_id' in product_sales.columns else 1
        
        # Obtener lista única de productos
        products = product_sales['producto_id'].unique()
        
        # Determinar rango de fechas
        if start_date is None:
            start_date = product_sales['fecha'].min()
            
        if end_date is None:
            end_date = product_sales['fecha'].max()
            
        # Asegurar que start_date y end_date sean datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Crear rango de fechas completo
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Crear todas las combinaciones producto-fecha
        product_date_combinations = pd.MultiIndex.from_product(
            [products, date_range], 
            names=['producto_id', 'fecha']
        )
        
        # Crear DataFrame con índice MultiIndex
        complete_df = pd.DataFrame(index=product_date_combinations).reset_index()
        
        # Asegurar que product_sales tiene fecha como datetime
        product_sales['fecha'] = pd.to_datetime(product_sales['fecha'])
        
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
        
        # Preservar categoria_id si existe
        if 'categoria_id' in product_sales.columns:
            # Crear mapeo producto_id -> categoria_id
            product_cats = product_sales.dropna(subset=['categoria_id'])
            product_cat_map = dict(zip(product_cats['producto_id'], product_cats['categoria_id']))
            
            # Aplicar mapeo
            result['categoria_id'] = result['producto_id'].map(product_cat_map)
        
        # Asegurar que tenant_id exista y tenga el valor correcto
        if 'tenant_id' not in result.columns:
            result['tenant_id'] = tenant_id
        
        return result
    
    def _generate_synthetic_data(self, start_date, end_date=None, tenant_id=1):
        """
        Genera datos sintéticos para desarrollo cuando no hay datos reales.
        
        Args:
            start_date (datetime): Fecha de inicio para los datos sintéticos
            end_date (datetime, optional): Fecha de fin para los datos sintéticos
            tenant_id (int): ID del tenant para el que generar datos
            
        Returns:
            pd.DataFrame: DataFrame con datos sintéticos
        """
        # Determinar fecha de fin si no se proporciona
        if end_date is None:
            end_date = datetime.now()
            
        # Convertir a datetime si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Crear rango de fechas
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Crear tendencia base (diferente por tenant)
        n_days = len(date_range)
        base_value = 100 + (tenant_id * 20)  # Diferentes valores base por tenant
        trend = np.linspace(base_value, base_value * 3, n_days) + np.random.normal(0, base_value * 0.1, n_days)
        
        # Añadir estacionalidad semanal
        weekday_effect = np.array([0.8, 0.9, 1.0, 1.0, 1.2, 1.5, 1.3])
        weekday_pattern = np.tile(weekday_effect, (n_days // 7) + 1)[:n_days]
        
        # Combinar elementos
        sales = trend * weekday_pattern
        
        # Crear DataFrame
        df = pd.DataFrame({
            'fecha': date_range,
            'total': sales,
            'cantidad': sales / 100,  # Cantidad aproximada
            'tenant_id': tenant_id  # Añadir tenant_id
        })
        
        self.logger.info(f"Datos sintéticos generados: {len(df)} días para tenant {tenant_id}")
        return df
    
    def _generate_synthetic_product_data(self, product_id=None, start_date=None, end_date=None, tenant_id=1):
        """
        Genera datos sintéticos por producto para desarrollo.
        
        Args:
            product_id (int, optional): ID de producto específico
            start_date (datetime, optional): Fecha de inicio
            end_date (datetime, optional): Fecha de fin
            tenant_id (int): ID del tenant para el que generar datos
            
        Returns:
            pd.DataFrame: DataFrame con datos sintéticos por producto
        """
        # Determinar fechas de inicio y fin
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
            
        if end_date is None:
            end_date = datetime.now()
            
        # Convertir a datetime si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)
        
        # Si no se especificó producto, generar datos para 5 productos
        if product_id is None:
            # Usar diferentes productos para diferentes tenants
            base_id = 100 + (tenant_id * 10)
            products = [base_id + 1, base_id + 2, base_id + 3, base_id + 4, base_id + 5]
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
                'cantidad': np.round(sales / (base * 0.1)),  # Cantidad aproximada
                'categoria_id': (prod % 5) + 1,  # Asignar categoría sintética
                'tenant_id': tenant_id  # Añadir tenant_id
            })
            
            all_data.append(prod_df)
        
        # Combinar todos los productos
        result = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Datos sintéticos de productos generados: {len(result)} registros para {len(products)} productos (tenant {tenant_id})")
        
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
        
        # Asegurar que fecha es datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features['fecha']):
            df_features['fecha'] = pd.to_datetime(df_features['fecha'])
        
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
        
        # Verificar si hay suficientes datos para dividir
        if len(X) < 2:
            self.logger.warning("No hay suficientes datos para crear secuencias de entrenamiento y validación")
            return X, y, np.array([]), np.array([])
        
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
            self.logger.warning(f"Datos insuficientes para producto {product_id}. Se necesitan al menos {min_required} registros, pero hay {len(product_data)}.")
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
        
        # Verificar si hay suficientes datos
        if len(X) < 2:
            self.logger.warning(f"No hay suficientes datos para producto {product_id}")
            return None, None, None, None
            
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
        # CORREGIDO: Aseguramos que la transformación inversa sea correcta
        # multiplicando por la desviación estándar y sumando la media
        if product_id is not None and product_id in self.product_scalers:
            # Usar scaler específico del producto
            scaler = self.product_scalers[product_id]
            # Obtener valores escalares para la media y desviación estándar
            mean_total = float(scaler['mean']['total'])
            std_total = max(1.0, float(scaler['std']['total']))  # Evitar std muy pequeñas
            
            # Aplicar transformación inversa
            return scaled_values * std_total + mean_total
        else:
            # Usar scaler general
            mean_total = float(self.mean['total'])
            std_total = max(1.0, float(self.std['total']))  # Evitar std muy pequeñas
            
            # Aplicar transformación inversa
            return scaled_values * std_total + mean_total
    
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
                self.logger.warning(f"Datos insuficientes para preparar secuencia del producto {product_id}")
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
    
    def get_product_info(self, product_id, tenant_id=1):
        """
        Obtiene información adicional de un producto.
        
        Args:
            product_id (int): ID del producto
            tenant_id (int): ID del tenant al que pertenece el producto
            
        Returns:
            dict: Información del producto (nombre, categoría, etc)
        """
        try:
            # MODIFICADO: Buscar primero por producto_id
            product_data = self.db_client.db[MONGO_PARAMS['collection_products']].find_one({
                "producto_id": product_id,
                "tenant_id": tenant_id  # Añadir tenant_id
            })
            
            if not product_data:
                # Si no se encuentra, intentar por _id
                product_data = self.db_client.db[MONGO_PARAMS['collection_products']].find_one({
                    "_id": product_id,
                    "tenant_id": tenant_id  # Añadir tenant_id
                })
                
            if not product_data:
                return {"nombre": f"Producto {product_id}", "categoria": "Desconocida"}
            
            # Identificar campos relevantes
            info = {
                "nombre": None,
                "categoria_id": None,
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
                    info["categoria_id"] = product_data[field]
                    # Buscar nombre de categoría si está disponible
                    if product_data[field] in self.category_info:
                        info["categoria"] = self.category_info[product_data[field]]["nombre"]
                    else:
                        info["categoria"] = f"Categoría {product_data[field]}"
                    break
            
            # Buscar precio
            for field in ["precio", "price", "precio_venta"]:
                if field in product_data:
                    info["precio"] = product_data[field]
                    break
            
            return info
        except Exception as e:
            self.logger.error(f"Error al obtener información del producto {product_id} para tenant {tenant_id}: {str(e)}")
            return {"nombre": f"Producto {product_id}", "categoria": "Desconocida"}
    
    def get_category_info(self, category_id, tenant_id=1):
        """
        Obtiene información de una categoría.
        
        Args:
            category_id: ID de la categoría
            tenant_id (int): ID del tenant al que pertenece la categoría
            
        Returns:
            dict: Información de la categoría
        """
        if category_id in self.category_info:
            return self.category_info[category_id]
        
        # Si no está en memoria, intentar obtener de la base de datos
        try:
            # Primero buscar en colección de categorías si existe
            category_collections = ["categorias", "raw_categoria", "raw_product_categories"]
            
            for collection in category_collections:
                if collection in self.db_client.db.list_collection_names():
                    category_data = self.db_client.db[collection].find_one({
                        "_id": category_id,
                        "tenant_id": tenant_id  # Añadir tenant_id
                    })
                    if category_data:
                        name_field = next((f for f in category_data.keys() if "nombre" in f.lower()), None)
                        if name_field:
                            return {
                                "nombre": category_data[name_field],
                                "productos": []  # No conocemos los productos aquí
                            }
            
            # Si no encontramos en colecciones de categorías, buscar productos con esta categoría
            productos = list(self.db_client.db[MONGO_PARAMS['collection_products']].find({
                "categoria_id": category_id,
                "tenant_id": tenant_id  # Añadir tenant_id
            }))
            
            if productos:
                # MODIFICADO: Usar producto_id para la lista de productos, no _id
                product_ids = []
                for p in productos:
                    if "producto_id" in p:
                        product_ids.append(p["producto_id"])
                    else:
                        product_ids.append(p["_id"])
                
                return {
                    "nombre": f"Categoría {category_id}",
                    "productos": product_ids
                }
                
            return {"nombre": f"Categoría {category_id}", "productos": []}
                
        except Exception as e:
            self.logger.error(f"Error al obtener información de categoría {category_id} para tenant {tenant_id}: {str(e)}")
            return {"nombre": f"Categoría {category_id}", "productos": []}
    
    def get_top_products(self, tenant_id=1, limit=10):
        """
        Obtiene los productos más vendidos basado en datos históricos.
        
        Args:
            tenant_id (int): ID del tenant para filtrar datos
            limit (int): Número de productos a retornar
                
        Returns:
            list: Lista de IDs de productos más vendidos
        """
        product_ids = []
        
        try:
            # ESTRATEGIA 1: Obtener productos desde raw_pedido_detalle (método principal)
            logging.info(f"Buscando top productos en raw_pedido_detalle para tenant {tenant_id}...")
            
            # Primero obtener pedidos para este tenant
            pedidos = list(self.db_client.db.raw_pedido.find(
                {"tenant_id": tenant_id},
                {"pedido_id": 1}
            ))
            
            if not pedidos:
                logging.info(f"No se encontraron pedidos para tenant {tenant_id}")
            else:
                # Extraer pedido_ids
                pedido_ids = [p.get("pedido_id") for p in pedidos if "pedido_id" in p]
                
                if pedido_ids:
                    # Pipeline de agregación para detalles de estos pedidos
                    pipeline = [
                        {"$match": {"pedido_id": {"$in": pedido_ids}}},
                        {"$group": {
                            "_id": "$producto_id",
                            "total_vendido": {"$sum": {"$ifNull": ["$total_linea", {"$multiply": ["$cantidad", "$precio_unitario"]}]}},
                            "cantidad_total": {"$sum": {"$ifNull": ["$cantidad", 0]}}
                        }},
                        {"$match": {
                            "total_vendido": {"$gt": 0}
                        }},
                        {"$sort": {"total_vendido": -1}},
                        {"$limit": limit}
                    ]
                    
                    top_products = list(self.db_client.db.raw_pedido_detalle.aggregate(pipeline))
                    
                    if top_products:
                        product_ids = [p["_id"] for p in top_products]
                        logging.info(f"Encontrados {len(product_ids)} productos en raw_pedido_detalle para tenant {tenant_id}")
                        return product_ids
            
            # ESTRATEGIA 2: Si no hay detalles, buscar directamente en raw_producto
            logging.info(f"Buscando en raw_producto para tenant {tenant_id}...")
            
            # Obtener los primeros productos de la colección de productos para este tenant
            productos = list(self.db_client.db.raw_producto.find({"tenant_id": tenant_id}).limit(limit))
            
            if productos:
                # MODIFICADO: Preferir producto_id sobre _id
                product_ids = []
                for p in productos:
                    if "producto_id" in p:
                        product_ids.append(p["producto_id"])
                    else:
                        product_ids.append(p["_id"])
                        
                logging.info(f"Encontrados {len(product_ids)} productos en raw_producto para tenant {tenant_id}")
                return product_ids
                
        except Exception as e:
            logging.error(f"Error al obtener productos más vendidos para tenant {tenant_id}: {str(e)}", exc_info=True)
            
        # ÚLTIMO RECURSO: Crear IDs de producto artificiales
        logging.warning(f"ADVERTENCIA: Usando IDs artificiales para tenant {tenant_id} debido a error en la consulta")
        base_id = 1000 + (tenant_id * 10)  # Diferentes IDs para diferentes tenants
        return [base_id + i for i in range(1, limit+1)]