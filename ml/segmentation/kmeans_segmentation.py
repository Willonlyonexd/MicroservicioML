import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_data_from_mongo(mongo_client, tenant_id):
    """
    Carga datos desde MongoDB para el tenant especificado
    
    Args:
        mongo_client: Cliente de MongoDB
        tenant_id: ID del tenant
        
    Returns:
        tuple: (df_clientes, df_cuenta_mesas, df_pedidos)
    """
    try:
        # Consultar datos de MongoDB específicos del tenant
        raw_clientes = list(mongo_client.db.raw_clientes.find({"tenant_id": tenant_id}))
        raw_cuenta_mesas = list(mongo_client.db.raw_cuenta_mesas.find({"tenant_id": tenant_id}))
        raw_pedidos = list(mongo_client.db.raw_pedidos.find({"tenant_id": tenant_id}))
        
        # Convertir a DataFrames
        df_clientes = pd.DataFrame(raw_clientes)
        df_cuenta_mesas = pd.DataFrame(raw_cuenta_mesas)
        df_pedidos = pd.DataFrame(raw_pedidos)
        
        # NUEVO: Filtrar cliente_id=1 (usado para ventas generales)
        if 'cliente_id' in df_clientes.columns:
            df_clientes = df_clientes[df_clientes['cliente_id'] != 1]
            logger.info(f"Filtrado cliente_id=1, quedan {len(df_clientes)} clientes")
        
        # Filtrar cuenta_mesas relacionadas con cliente_id=1
        if 'cliente_id' in df_cuenta_mesas.columns:
            df_cuenta_mesas = df_cuenta_mesas[df_cuenta_mesas['cliente_id'] != 1]
            logger.info(f"Filtradas mesas de cliente_id=1, quedan {len(df_cuenta_mesas)} mesas")
        
        return df_clientes, df_cuenta_mesas, df_pedidos
    except Exception as e:
        logger.error(f"Error cargando datos desde MongoDB: {str(e)}")
        return None, None, None

def prepare_features(df_clientes, df_cuenta_mesas, df_pedidos):
    """
    Prepara las características para la segmentación
    
    Args:
        df_clientes: DataFrame con datos de clientes
        df_cuenta_mesas: DataFrame con datos de mesas
        df_pedidos: DataFrame con datos de pedidos
        
    Returns:
        DataFrame: Características para clustering
    """
    # 1. Unir cuenta_mesas con pedidos
    cuenta_mesas_con_pedidos = pd.merge(
        df_cuenta_mesas[['cliente_id', 'cuenta_mesa_id', 'num_comensales']], 
        df_pedidos[['cuenta_mesa_id', 'total']], 
        on='cuenta_mesa_id', 
        how='inner'
    )
    
    # 2. Calcular estadísticas por cliente
    # Número de visitas
    visitas_por_cliente = df_cuenta_mesas.groupby('cliente_id').size().reset_index(name='num_visitas')
    
    # Gasto total
    gasto_total = cuenta_mesas_con_pedidos.groupby('cliente_id')['total'].sum().reset_index()
    gasto_total.columns = ['cliente_id', 'gasto_total']
    
    # Gasto promedio por visita
    gasto_promedio = pd.merge(gasto_total, visitas_por_cliente, on='cliente_id')
    gasto_promedio['gasto_promedio_visita'] = gasto_promedio['gasto_total'] / gasto_promedio['num_visitas']
    
    # Comensales promedio
    comensales_promedio = df_cuenta_mesas.groupby('cliente_id')['num_comensales'].mean().reset_index()
    comensales_promedio.columns = ['cliente_id', 'comensales_promedio']
    
    # 3. Combinar todas las características
    features = pd.merge(visitas_por_cliente, gasto_total, on='cliente_id')
    features = pd.merge(features, gasto_promedio[['cliente_id', 'gasto_promedio_visita']], on='cliente_id')
    features = pd.merge(features, comensales_promedio, on='cliente_id')
    
    # 4. Unir con datos básicos de clientes
    if 'nombre' in df_clientes.columns and 'apellido' in df_clientes.columns:
        clientes_info = df_clientes[['cliente_id', 'nombre', 'apellido']]
        features = pd.merge(features, clientes_info, on='cliente_id')
    
    # 5. Eliminar valores nulos
    features.fillna({
        'num_visitas': 0,
        'gasto_total': 0,
        'gasto_promedio_visita': 0,
        'comensales_promedio': 0
    }, inplace=True)
    
    return features

def run_kmeans(features, n_clusters=4):  # MODIFICADO: Ahora 4 clusters por defecto
    """
    Ejecuta el algoritmo K-means
    
    Args:
        features: DataFrame con características
        n_clusters: Número de clusters (ahora 4 por defecto)
        
    Returns:
        tuple: (datos_segmentados, modelo_kmeans, scaler)
    """
    # 1. Copiar datos originales
    df = features.copy()
    
    # 2. Seleccionar características para clustering
    columns_for_clustering = ['num_visitas', 'gasto_total', 'gasto_promedio_visita', 'comensales_promedio']
    X = df[columns_for_clustering]
    
    # 3. Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Ejecutar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 5. Calcular métricas adicionales
    df['distancia_al_centroide'] = np.min(kmeans.transform(X_scaled), axis=1)
    
    return df, kmeans, scaler

def visualize_clusters(df, output_dir):
    """
    Genera visualizaciones de los clusters
    
    Args:
        df: DataFrame con datos segmentados
        output_dir: Directorio donde guardar las visualizaciones
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribución de clusters
    plt.figure(figsize=(10, 6))
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax = cluster_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribución de Clientes por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Número de Clientes')
    
    # Añadir etiquetas con el número exacto
    for i, v in enumerate(cluster_counts):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
    plt.close()
    
    # 2. Gráfico de dispersión: Gasto Total vs Número de Visitas
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['gasto_total'], df['num_visitas'], 
                         c=df['cluster'], cmap='viridis', 
                         s=100, alpha=0.7)
    plt.title('Segmentación de Clientes: Gasto Total vs Número de Visitas')
    plt.xlabel('Gasto Total')
    plt.ylabel('Número de Visitas')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gasto_vs_visitas.png'))
    plt.close()
    
    # 3. Gráfico de dispersión: Gasto Promedio vs Comensales Promedio
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['gasto_promedio_visita'], df['comensales_promedio'], 
                         c=df['cluster'], cmap='viridis', 
                         s=100, alpha=0.7)
    plt.title('Segmentación de Clientes: Gasto Promedio vs Comensales Promedio')
    plt.xlabel('Gasto Promedio por Visita')
    plt.ylabel('Comensales Promedio')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gasto_promedio_vs_comensales.png'))
    plt.close()

def save_results(df, kmeans_model, scaler, output_dir):
    """
    Guarda el modelo y los resultados de la segmentación
    
    Args:
        df: DataFrame con datos segmentados
        kmeans_model: Modelo de K-means entrenado
        scaler: Scaler usado para normalizar datos
        output_dir: Directorio donde guardar los resultados
        
    Returns:
        dict: Mapeo de cluster_id a nombre
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Guardar modelo K-means
    joblib.dump(kmeans_model, os.path.join(output_dir, 'kmeans_model.pkl'))
    
    # 2. Guardar scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # 3. Guardar datos segmentados
    df.to_csv(os.path.join(output_dir, 'segmented_customers.csv'), index=False)
    
    # 4. Determinar nombres de clusters basados en comportamiento
    cluster_stats = df.groupby('cluster').agg({
        'num_visitas': 'mean',
        'gasto_total': 'mean',
        'gasto_promedio_visita': 'mean',
        'comensales_promedio': 'mean'
    }).reset_index()
    
    # Ordenar clusters por gasto total para asignar nombres coherentes
    cluster_stats = cluster_stats.sort_values(by='gasto_total', ascending=False)
    
    # MODIFICADO: Asignar nombres a 4 clusters
    cluster_names = {}
    for i, row in cluster_stats.iterrows():
        cluster_id = int(row['cluster'])
        if i == 0:
            cluster_names[cluster_id] = "VIP"       # NUEVO: Nivel más alto
        elif i == 1:
            cluster_names[cluster_id] = "PREMIUM"
        elif i == 2:
            cluster_names[cluster_id] = "REGULAR"
        else:
            cluster_names[cluster_id] = "OCASIONAL"
    
    # Guardar mapeo de nombres
    with open(os.path.join(output_dir, 'cluster_names.txt'), 'w') as f:
        for cluster_id, name in cluster_names.items():
            f.write(f"{cluster_id}: {name}\n")
    
    # Guardar también como JSON para facilitar su carga
    with open(os.path.join(output_dir, 'cluster_names.json'), 'w') as f:
        json.dump(cluster_names, f)
    
    return cluster_names