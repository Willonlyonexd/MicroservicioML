from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de MongoDB
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "microserviceMl"  # Nombre exacto de la base de datos

def crear_segmentacion():
    """Script principal para crear segmentación de clientes en 4 segmentos"""
    try:
        # 1. CONECTAR A MONGODB
        logger.info("Conectando a MongoDB...")
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        
        # 2. VERIFICAR COLECCIONES DISPONIBLES
        collections = db.list_collection_names()
        logger.info(f"Colecciones disponibles: {collections}")
        
        # 3. EXTRAER DATOS
        logger.info("Extrayendo datos de raw_clientes, raw_cuenta_mesas y raw_pedidos...")
        
        clientes_raw = list(db.raw_clientes.find())
        mesas_raw = list(db.raw_cuenta_mesas.find())
        pedidos_raw = list(db.raw_pedidos.find())
        
        logger.info(f"Datos encontrados: {len(clientes_raw)} clientes, {len(mesas_raw)} mesas, {len(pedidos_raw)} pedidos")
        
        # 4. CONVERTIR A DATAFRAMES
        clientes_df = pd.DataFrame(clientes_raw)
        mesas_df = pd.DataFrame(mesas_raw)
        pedidos_df = pd.DataFrame(pedidos_raw)
        
        # Eliminar el campo _id y filtrar cliente_id=1
        if '_id' in clientes_df.columns:
            clientes_df = clientes_df.drop('_id', axis=1)
        if '_id' in mesas_df.columns:
            mesas_df = mesas_df.drop('_id', axis=1)
        if '_id' in pedidos_df.columns:
            pedidos_df = pedidos_df.drop('_id', axis=1)
        
        # NUEVO: Filtrar cliente_id=1 (ventas generales de usuarios no registrados)
        clientes_df = clientes_df[clientes_df['cliente_id'] != 1]
        logger.info(f"Excluyendo cliente_id=1. Procesando {len(clientes_df)} clientes restantes")
            
        # 5. CALCULAR MÉTRICAS PARA SEGMENTACIÓN
        features_df = calcular_metricas_clientes(clientes_df, mesas_df, pedidos_df)
        
        # 6. SEGMENTAR CON K-MEANS (ahora 4 clusters)
        logger.info(f"Ejecutando K-means con {len(features_df)} clientes y 4 segmentos")
        segmented_data, cluster_info = ejecutar_kmeans(features_df, n_clusters=4)
        
        # 7. GUARDAR RESULTADOS
        logger.info("Guardando resultados en MongoDB...")
        
        # Limpiar colecciones existentes
        db.cliente_segmentacion.delete_many({})
        db.cluster_perfiles.delete_many({})
        
        # Guardar perfiles de clusters
        for cluster in cluster_info:
            db.cluster_perfiles.insert_one(cluster)
            logger.info(f"Cluster guardado: {cluster['nombre']} con {cluster['num_clientes']} clientes")
        
        # Guardar datos de clientes segmentados (en lotes para evitar problemas de memoria)
        batch_size = 100
        for i in range(0, len(segmented_data), batch_size):
            batch = segmented_data[i:i+batch_size]
            db.cliente_segmentacion.insert_many(batch)
            logger.info(f"Guardados clientes {i+1} a {i+len(batch)}")
        
        # Verificar que los datos se guardaron correctamente
        cliente_count = db.cliente_segmentacion.count_documents({})
        cluster_count = db.cluster_perfiles.count_documents({})
        
        logger.info(f"Verificación final: {cliente_count} clientes y {cluster_count} clusters guardados")
        
        # Mostrar resumen
        print("\n=== RESUMEN DE SEGMENTACIÓN ===")
        for cluster in cluster_info:
            print(f"• {cluster['nombre']}: {cluster['num_clientes']} clientes, "
                  f"Gasto: ${cluster['gasto_promedio']:.2f}, "
                  f"Visitas: {cluster['visitas_promedio']:.1f}")
        
        # Cerrar conexión
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error en la segmentación: {str(e)}", exc_info=True)
        print(f"❌ ERROR: {str(e)}")
        return False

def calcular_metricas_clientes(clientes_df, mesas_df, pedidos_df):
    """Calcula métricas de cliente basadas en datos reales"""
    try:
        # Inicializar dataframe de características
        features = clientes_df[['cliente_id']].copy()
        logger.info(f"Procesando {len(features)} clientes")
        
        # 1. CALCULAR NÚMERO DE VISITAS
        # Contar cuántas veces aparece cada cliente_id en cuenta_mesas
        visitas = mesas_df.groupby('cliente_id').size().reset_index(name='num_visitas')
        features = features.merge(visitas, on='cliente_id', how='left')
        features['num_visitas'] = features['num_visitas'].fillna(0).astype(int)
        
        # 2. CALCULAR GASTO TOTAL
        # Primero unir pedidos con cuenta_mesas para obtener el cliente_id
        pedidos_con_cliente = pd.merge(
            pedidos_df, 
            mesas_df[['cuenta_mesa_id', 'cliente_id']], 
            on='cuenta_mesa_id', 
            how='inner'
        )
        
        # Excluir explícitamente cliente_id=1
        pedidos_con_cliente = pedidos_con_cliente[pedidos_con_cliente['cliente_id'] != 1]
        
        # Sumar el total de cada cliente
        if 'total' in pedidos_con_cliente.columns:
            gasto = pedidos_con_cliente.groupby('cliente_id')['total'].sum().reset_index(name='gasto_total')
            features = features.merge(gasto, on='cliente_id', how='left')
        else:
            features['gasto_total'] = 0.0
        
        features['gasto_total'] = features['gasto_total'].fillna(0.0)
        
        # 3. CALCULAR PROMEDIO DE COMENSALES
        # Filtrar mesas para excluir cliente_id=1
        mesas_filtered = mesas_df[mesas_df['cliente_id'] != 1]
        
        # Obtener el promedio de comensales por cliente
        if 'num_comensales' in mesas_filtered.columns:
            comensales = mesas_filtered.groupby('cliente_id')['num_comensales'].mean().reset_index(name='comensales_promedio')
            features = features.merge(comensales, on='cliente_id', how='left')
        else:
            features['comensales_promedio'] = 2.0  # Valor por defecto
        
        features['comensales_promedio'] = features['comensales_promedio'].fillna(2.0)
        
        # 4. FILTRAR CLIENTES SIN ACTIVIDAD
        # Solo considerar clientes con al menos una visita
        active_features = features[features['num_visitas'] > 0].reset_index(drop=True)
        
        # Si hay muy pocos clientes activos, usar todos
        if len(active_features) < 20:
            logger.warning(f"Solo {len(active_features)} clientes activos. Usando todos los clientes")
            return features
        
        logger.info(f"Se calcularon métricas para {len(active_features)} clientes activos")
        return active_features
        
    except Exception as e:
        logger.error(f"Error al calcular métricas: {str(e)}")
        raise

def ejecutar_kmeans(features_df, n_clusters=4):
    """Ejecuta el algoritmo K-means para segmentar clientes en 4 niveles"""
    # Seleccionar características para clustering
    X = features_df[['num_visitas', 'gasto_total', 'comensales_promedio']].copy()
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ejecutar K-means con 4 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Añadir cluster al DataFrame
    features_df['cluster'] = clusters
    
    # Determinar nombres de clusters basados en gasto total promedio
    cluster_stats = features_df.groupby('cluster').agg({
        'gasto_total': 'mean',
        'num_visitas': 'mean',
        'comensales_promedio': 'mean',
        'cliente_id': 'count'
    }).reset_index()
    
    # Ordenar clusters por gasto promedio (de mayor a menor)
    cluster_stats = cluster_stats.sort_values(by='gasto_total', ascending=False)
    
    # Asignar nombres a clusters (ahora 4 niveles)
    cluster_names = {}
    for i, row in cluster_stats.iterrows():
        if i == 0:
            cluster_names[int(row['cluster'])] = "VIP"  # El más alto
        elif i == 1:
            cluster_names[int(row['cluster'])] = "PREMIUM"
        elif i == 2:
            cluster_names[int(row['cluster'])] = "REGULAR"
        else:
            cluster_names[int(row['cluster'])] = "OCASIONAL"
    
    # Crear documentos para MongoDB
    
    # 1. Perfiles de clusters
    cluster_docs = []
    for cluster_id, name in cluster_names.items():
        stats = cluster_stats[cluster_stats['cluster'] == cluster_id].iloc[0]
        
        cluster_docs.append({
            "cluster_id": int(cluster_id),
            "nombre": name,
            "num_clientes": int(stats['cliente_id']),
            "gasto_promedio": float(stats['gasto_total']),
            "visitas_promedio": float(stats['num_visitas']),
            "comensales_promedio": float(stats['comensales_promedio']),
            "descripcion": get_cluster_description(name),
            "tenant_id": 1,
            "timestamp": datetime.now()
        })
    
    # 2. Datos de clientes segmentados
    client_docs = []
    for _, row in features_df.iterrows():
        cluster_id = int(row['cluster'])
        client_docs.append({
            "cliente_id": int(row['cliente_id']),
            "cluster": cluster_id,
            "cluster_nombre": cluster_names.get(cluster_id, "DESCONOCIDO"),
            "num_visitas": int(row['num_visitas']),
            "comensales_promedio": float(row['comensales_promedio']),
            "gasto_total": float(row['gasto_total']),
            "tenant_id": 1,
            "timestamp": datetime.now()
        })
    
    return client_docs, cluster_docs

def get_cluster_description(name):
    """Devuelve una descripción para cada tipo de cluster"""
    if name == "VIP":
        return "Clientes exclusivos con alto valor, gasto excepcional y visitas muy frecuentes"
    elif name == "PREMIUM":
        return "Clientes de alto valor con gasto elevado y visitas frecuentes"
    elif name == "REGULAR":
        return "Clientes con nivel de gasto medio y frecuencia moderada"
    else:
        return "Clientes ocasionales con baja frecuencia y gasto reducido"

if __name__ == "__main__":
    print("=== SEGMENTACIÓN DE CLIENTES EN 4 NIVELES ===")
    print("Niveles: VIP, PREMIUM, REGULAR, OCASIONAL")
    print(f"Conectando a la base de datos: {DB_NAME}")
    if crear_segmentacion():
        print("\n✅ Segmentación con datos reales completada con éxito")
        print("Datos guardados en las colecciones:")
        print("  - cliente_segmentacion")
        print("  - cluster_perfiles")
        print("\nAhora puedes usar las consultas GraphQL:")
        print("  - clientesSegmentados")
        print("  - todosClusters")
        print("  - segmentacionGeneral")
    else:
        print("\n❌ Error al ejecutar la segmentación con datos reales")