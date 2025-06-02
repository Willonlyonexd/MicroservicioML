import pandas as pd
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

def enrich_customer_data(clientes_df, cuenta_mesas_df, pedidos_df):
    """
    Enriquece los datos de clientes con información adicional de sus comportamientos
    
    Args:
        clientes_df: DataFrame con datos de clientes
        cuenta_mesas_df: DataFrame con datos de cuenta_mesas
        pedidos_df: DataFrame con datos de pedidos
    
    Returns:
        DataFrame: Datos de clientes enriquecidos
    """
    # Asegurarse de que los DataFrames tienen los campos necesarios
    required_fields = {
        'clientes_df': ['cliente_id', 'nombre', 'apellido'],
        'cuenta_mesas_df': ['cliente_id', 'cuenta_mesa_id', 'num_comensales', 'fecha'],
        'pedidos_df': ['cuenta_mesa_id', 'total', 'fecha']
    }
    
    for df_name, fields in required_fields.items():
        df = locals()[df_name]
        if df is None or df.empty:
            logger.error(f"Error: {df_name} está vacío o es None")
            return None
        for field in fields:
            if field not in df.columns:
                logger.error(f"Error: Campo '{field}' no encontrado en {df_name}")
                return None
    
    # NUEVO: Filtrar cliente_id=1 (ventas generales)
    clientes_df = clientes_df[clientes_df['cliente_id'] != 1]
    cuenta_mesas_df = cuenta_mesas_df[cuenta_mesas_df['cliente_id'] != 1]
    
    # 1. Calcular frecuencia de visitas
    if 'fecha' in cuenta_mesas_df.columns:
        # Convertir fechas si son strings
        if cuenta_mesas_df['fecha'].dtype == 'object':
            try:
                cuenta_mesas_df['fecha'] = pd.to_datetime(cuenta_mesas_df['fecha'])
            except:
                logger.error("Error al convertir fechas en cuenta_mesas_df")
        
        # Calcular días desde la primera hasta la última visita
        visitas_por_cliente = cuenta_mesas_df.groupby('cliente_id')['fecha'].agg(['min', 'max', 'count'])
        visitas_por_cliente.columns = ['primera_visita', 'ultima_visita', 'num_visitas']
        
        # Calcular días entre primera y última visita
        visitas_por_cliente['dias_entre_visitas'] = (visitas_por_cliente['ultima_visita'] - visitas_por_cliente['primera_visita']).dt.days
        
        # Calcular frecuencia (visitas por mes)
        visitas_por_cliente['frecuencia_mensual'] = np.where(
            visitas_por_cliente['dias_entre_visitas'] > 0,
            visitas_por_cliente['num_visitas'] * 30 / visitas_por_cliente['dias_entre_visitas'],
            0
        )
    else:
        visitas_por_cliente = cuenta_mesas_df.groupby('cliente_id').size().reset_index(name='num_visitas')
        visitas_por_cliente['frecuencia_mensual'] = visitas_por_cliente['num_visitas'] / 3  # Asumiendo 3 meses de datos
    
    # 2. Calcular gasto promedio por visita
    cuenta_mesas_con_pedidos = pd.merge(
        cuenta_mesas_df[['cliente_id', 'cuenta_mesa_id']], 
        pedidos_df[['cuenta_mesa_id', 'total']], 
        on='cuenta_mesa_id', 
        how='inner'
    )
    
    gasto_total_por_cliente = cuenta_mesas_con_pedidos.groupby('cliente_id')['total'].sum().reset_index()
    gasto_total_por_cliente.columns = ['cliente_id', 'gasto_total']
    
    # 3. Calcular comensales promedio
    comensales_promedio = cuenta_mesas_df.groupby('cliente_id')['num_comensales'].mean().reset_index()
    comensales_promedio.columns = ['cliente_id', 'comensales_promedio']
    
    # 4. Unir toda la información
    result = clientes_df.merge(visitas_por_cliente, on='cliente_id', how='left')
    result = result.merge(gasto_total_por_cliente, on='cliente_id', how='left')
    result = result.merge(comensales_promedio, on='cliente_id', how='left')
    
    # 5. Calcular gasto promedio por visita
    result['gasto_promedio_visita'] = result['gasto_total'] / result['num_visitas']
    
    # 6. Calcular lifetime value (LTV) - simple
    # Asumimos un LTV de 12 meses basado en gasto mensual
    result['ltv_12m'] = result['gasto_total'] * (12 / 3)  # Asumiendo 3 meses de datos históricos
    
    # 7. Rellenar valores nulos
    result.fillna({
        'num_visitas': 0,
        'gasto_total': 0,
        'comensales_promedio': 0,
        'gasto_promedio_visita': 0,
        'frecuencia_mensual': 0,
        'ltv_12m': 0
    }, inplace=True)
    
    return result

def prepare_mongodb_data(segmented_data, cluster_names, tenant_id=1):
    """
    Prepara los datos para guardar en MongoDB
    
    Args:
        segmented_data: DataFrame con datos segmentados
        cluster_names: Dict con mapeo de cluster_id a nombre
        tenant_id: ID del tenant
        
    Returns:
        tuple: (documentos_clientes, documentos_clusters)
    """
    # 1. Preparar documentos de clientes
    client_docs = []
    for _, row in segmented_data.iterrows():
        # NUEVO: Excluir cliente_id=1 por seguridad
        if row['cliente_id'] == 1:
            continue
            
        cluster_id = int(row['cluster'])
        doc = {
            "cliente_id": int(row['cliente_id']),
            "cluster": cluster_id,
            "cluster_nombre": cluster_names.get(cluster_id, "DESCONOCIDO"),
            "num_visitas": int(row['num_visitas']),
            "comensales_promedio": float(row['comensales_promedio']),
            "gasto_total": float(row['gasto_total']),
            "timestamp": datetime.now(),
            "tenant_id": tenant_id
        }
        client_docs.append(doc)
    
    # 2. Preparar documentos de perfiles de cluster
    cluster_docs = []
    for cluster_id, name in cluster_names.items():
        cluster_data = segmented_data[segmented_data['cluster'] == cluster_id]
        
        if not cluster_data.empty:
            profile = {
                "cluster_id": int(cluster_id),
                "nombre": name,
                "num_clientes": len(cluster_data),
                "gasto_promedio": float(cluster_data['gasto_total'].mean()),
                "visitas_promedio": float(cluster_data['num_visitas'].mean()),
                "comensales_promedio": float(cluster_data['comensales_promedio'].mean()),
                "tenant_id": tenant_id,
                "timestamp": datetime.now()
            }
            
            # MODIFICADO: Determinar descripción basada en el nombre del cluster
            if name == "VIP":
                profile["descripcion"] = "Clientes exclusivos con alto valor, gasto excepcional y visitas muy frecuentes"
            elif name == "PREMIUM":
                profile["descripcion"] = "Clientes de alto valor con gasto elevado y visitas frecuentes"
            elif name == "REGULAR":
                profile["descripcion"] = "Clientes con nivel de gasto medio y frecuencia moderada"
            else:  # OCASIONAL
                profile["descripcion"] = "Clientes ocasionales con baja frecuencia y gasto reducido"
                
            cluster_docs.append(profile)
    
    return client_docs, cluster_docs

def save_to_mongodb(mongo_client, client_docs, cluster_docs):
    """
    Guarda los documentos en MongoDB
    
    Args:
        mongo_client: Cliente de MongoDB
        client_docs: Lista de documentos de clientes
        cluster_docs: Lista de documentos de clusters
        
    Returns:
        bool: True si se guardaron correctamente, False en caso contrario
    """
    try:
        # Asegúrate de que tengas documentos para guardar
        if not client_docs or not cluster_docs:
            logger.warning("No hay documentos para guardar en MongoDB")
            return False

        # 1. Guardar clientes segmentados
        if client_docs:
            tenant_id = client_docs[0].get('tenant_id', 1)
            
            # Eliminar documentos anteriores del mismo tenant - Usando syntax más simple
            try:
                mongo_client.db.cliente_segmentacion.delete_many({"tenant_id": tenant_id})
            except Exception as e:
                logger.error(f"Error al eliminar clientes anteriores: {str(e)}")
                return False
            
            # Insertar documentos uno por uno en lugar de en bloque
            success_count = 0
            for doc in client_docs:
                try:
                    mongo_client.db.cliente_segmentacion.insert_one(doc)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error al insertar cliente {doc.get('cliente_id')}: {str(e)}")
            
            logger.info(f"Guardados {success_count} de {len(client_docs)} documentos de clientes en MongoDB")
        
        # 2. Guardar perfiles de clusters
        if cluster_docs:
            tenant_id = cluster_docs[0].get('tenant_id', 1)
            
            # Eliminar documentos anteriores del mismo tenant
            try:
                mongo_client.db.cluster_perfiles.delete_many({"tenant_id": tenant_id})
            except Exception as e:
                logger.error(f"Error al eliminar perfiles de cluster anteriores: {str(e)}")
                return False
            
            # Insertar documentos uno por uno
            success_count = 0
            for doc in cluster_docs:
                try:
                    mongo_client.db.cluster_perfiles.insert_one(doc)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error al insertar cluster {doc.get('cluster_id')}: {str(e)}")
            
            logger.info(f"Guardados {success_count} de {len(cluster_docs)} perfiles de clusters en MongoDB")
        
        return True
    except Exception as e:
        logger.error(f"Error al guardar en MongoDB: {str(e)}")
        return False