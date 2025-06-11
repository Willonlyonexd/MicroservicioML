import pandas as pd
import numpy as np
from datetime import datetime
import logging

class RecommendationModel:
    def __init__(self, data_manager):
        """Inicializa el modelo de recomendaciones"""
        self.data_manager = data_manager
        self.user_profiles = {}
        self.product_similarities = {}
        self.logger = logging.getLogger('recommendation_model')
    
    def get_personal_recommendations(self, client_id, tenant_id, limit=5, 
                                    exclude_products=None, category_ids=None, 
                                    context=None):
        """Genera recomendaciones personalizadas para un cliente"""
        try:
            # Obtener historial de compras
            purchase_history = self.data_manager.get_client_purchase_history(client_id, tenant_id)
            
            if purchase_history.empty:
                self.logger.info(f"No hay historial de compras para cliente {client_id}, usando recomendaciones generales")
                # Si no hay historial, usar recomendaciones generales
                return self.get_general_recommendations(tenant_id, limit, context)
                
            # Productos ya comprados (para excluir)
            bought_products = set(purchase_history['producto_id'].astype(str).tolist())
            if exclude_products:
                bought_products.update(exclude_products)
                
            # Obtener productos populares
            popular_products = self.data_manager.get_product_popularity(tenant_id, context)
            
            # Obtener productos comprados juntos
            similar_products = pd.DataFrame()
            for product_id in purchase_history['producto_id'].head(10):
                similar = self._get_similar_products(product_id, tenant_id)
                similar_products = pd.concat([similar_products, similar])
                
            # Normalizar puntuaciones
            if not similar_products.empty:
                # Asegurar que todas las columnas necesarias existen
                required_columns = ['producto_id', 'score', 'nombre', 'precio']
                for col in required_columns:
                    if col not in similar_products.columns:
                        similar_products[col] = np.nan
                
                # Si no existe la columna categoria_id, agrégala con valor por defecto
                if 'categoria_id' not in similar_products.columns:
                    similar_products['categoria_id'] = np.nan
                
                # Agrupar solo por columnas que existen
                agg_dict = {
                    'score': 'mean', 
                    'nombre': 'first', 
                    'precio': 'first'
                }
                
                if 'categoria_id' in similar_products.columns:
                    agg_dict['categoria_id'] = 'first'
                
                similar_products = similar_products.groupby('producto_id').agg(agg_dict).reset_index()
                
            # Crear puntuaciones combinadas
            if similar_products.empty:
                all_candidates = popular_products.copy()
                if not all_candidates.empty:
                    all_candidates['score'] = all_candidates['pedidos_totales'] / all_candidates['pedidos_totales'].max()
                    all_candidates['reason'] = 'Popular entre nuestros clientes'
                    all_candidates['final_score'] = all_candidates['score']
            else:
                # Asegurar que popular_products tiene las columnas necesarias
                for col in ['producto_id']:
                    if col not in popular_products.columns:
                        self.logger.warning(f"Columna {col} no encontrada en popular_products")
                        popular_products[col] = np.nan
                
                # Realizar merge con manejo seguro
                try:
                    all_candidates = pd.merge(
                        popular_products,
                        similar_products,
                        on='producto_id',
                        how='outer',
                        suffixes=('_pop', '_sim')
                    )
                except Exception as e:
                    self.logger.error(f"Error en merge de dataframes: {str(e)}")
                    self.logger.info(f"Popular columns: {popular_products.columns.tolist()}")
                    self.logger.info(f"Similar columns: {similar_products.columns.tolist()}")
                    # Fallback a usar solo productos populares
                    all_candidates = popular_products.copy()
                
                # Normalizar y combinar puntuaciones
                if 'pedidos_totales' in all_candidates.columns:
                    max_pop = all_candidates['pedidos_totales'].max()
                    all_candidates['pop_score'] = all_candidates['pedidos_totales'] / max_pop if max_pop > 0 else 0
                else:
                    all_candidates['pop_score'] = 0
                    
                if 'score' in all_candidates.columns:
                    all_candidates['score'] = all_candidates['score'].fillna(0)
                else:
                    all_candidates['score'] = 0
                
                # Combinar puntuaciones (70% similitud, 30% popularidad)
                all_candidates['final_score'] = (all_candidates['score'] * 0.7 + 
                                                all_candidates['pop_score'] * 0.3)
                
                # Generar razones para las recomendaciones
                self._add_recommendation_reasons(all_candidates, purchase_history, context)
            
            # Filtrar por categoría si se especifica
            if category_ids and 'categoria_id' in all_candidates.columns:
                # Manejar valores nulos y convertir a string
                all_candidates['categoria_id'] = all_candidates['categoria_id'].fillna('').astype(str)
                all_candidates = all_candidates[all_candidates['categoria_id'].isin([str(c) for c in category_ids])]
                
            # Excluir productos ya comprados
            if bought_products and 'producto_id' in all_candidates.columns:
                all_candidates = all_candidates[~all_candidates['producto_id'].astype(str).isin(bought_products)]
                
            # Verificar si hay candidatos después de los filtros
            if all_candidates.empty:
                self.logger.warning("No hay candidatos después de aplicar filtros, regresando lista vacía")
                return []
                
            # Ordenar por puntuación final y tomar los primeros 'limit'
            score_col = 'final_score' if 'final_score' in all_candidates.columns else 'score'
            if score_col in all_candidates.columns:
                recommendations = all_candidates.sort_values(
                    by=score_col, 
                    ascending=False
                ).head(limit)
            else:
                self.logger.warning(f"Columna de puntuación {score_col} no encontrada, usando orden original")
                recommendations = all_candidates.head(limit)
            
            return self._format_recommendations(recommendations, tenant_id)
            
        except Exception as e:
            self.logger.error(f"Error generando recomendaciones personales: {str(e)}", exc_info=True)
            # Intentar fallback a recomendaciones generales
            self.logger.info("Intentando fallback a recomendaciones generales")
            try:
                return self.get_general_recommendations(tenant_id, limit, context)
            except Exception as e2:
                self.logger.error(f"Error también en fallback: {str(e2)}")
                return []
    
    def get_general_recommendations(self, tenant_id, limit=5, context=None, category_ids=None):
        """Genera recomendaciones generales basadas en popularidad"""
        try:
            # Obtener productos populares
            popular_products = self.data_manager.get_product_popularity(tenant_id, context)
            
            if popular_products.empty:
                self.logger.warning(f"No se encontraron productos populares para tenant {tenant_id}")
                return []
                
            # Normalizar puntuación por popularidad
            max_pop = popular_products['pedidos_totales'].max()
            popular_products['score'] = popular_products['pedidos_totales'] / max_pop if max_pop > 0 else 0
            
            # Añadir razones contextuales
            popular_products['reason'] = 'Popular entre nuestros clientes'
            
            if context:
                time_reasons = {
                    'MORNING': 'Popular en desayunos',
                    'LUNCH': 'Favorito para el almuerzo',
                    'AFTERNOON': 'Ideal para la tarde',
                    'DINNER': 'Muy pedido en cenas'
                }
                
                day_reasons = {
                    0: 'Popular los domingos',
                    1: 'Favorito los lunes',
                    2: 'Tendencia los martes',
                    3: 'Muy pedido los miércoles',
                    4: 'Popular los jueves',
                    5: 'Favorito los viernes',
                    6: 'Tendencia los sábados'
                }
                
                if 'timeOfDay' in context and context['timeOfDay'] in time_reasons:
                    # Aplicar razón por hora del día a algunos productos
                    if len(popular_products) > 0:
                        idx_range = min(3, len(popular_products))
                        popular_products.iloc[:idx_range, popular_products.columns.get_loc('reason')] = time_reasons[context['timeOfDay']]
                    
                if 'dayOfWeek' in context and context['dayOfWeek'] in day_reasons:
                    # Alternar razones para diversificar
                    if len(popular_products) > 3:
                        idx_range = min(3, len(popular_products) - 3)
                        popular_products.iloc[3:3+idx_range, popular_products.columns.get_loc('reason')] = day_reasons[context['dayOfWeek']]
            
            # Filtrar por categoría si se especifica
            if category_ids and 'categoria_id' in popular_products.columns:
                # Manejar valores nulos y convertir a string
                popular_products['categoria_id'] = popular_products['categoria_id'].fillna('').astype(str)
                filtered = popular_products[popular_products['categoria_id'].isin([str(c) for c in category_ids])]
                # Solo usar filtro si produce resultados, de lo contrario mantener todos
                if not filtered.empty:
                    popular_products = filtered
            
            # Limitar a la cantidad solicitada
            recommendations = popular_products.head(limit)
            
            return self._format_recommendations(recommendations, tenant_id)
            
        except Exception as e:
            self.logger.error(f"Error generando recomendaciones generales: {str(e)}", exc_info=True)
            return []
    
    def _get_similar_products(self, product_id, tenant_id):
        """Encuentra productos similares a uno dado"""
        # Verificar si ya tenemos los productos similares en caché
        cache_key = f"{tenant_id}_{product_id}"
        if cache_key in self.product_similarities:
            return self.product_similarities[cache_key]
        
        # Obtener productos comprados juntos
        bought_together = self.data_manager.get_products_bought_together(tenant_id)
        
        if bought_together.empty:
            return pd.DataFrame()
        
        # Filtrar por producto de interés
        product_pairs = bought_together[
            (bought_together['producto1_id'] == product_id) | 
            (bought_together['producto2_id'] == product_id)
        ]
        
        if product_pairs.empty:
            return pd.DataFrame()
        
        # Extraer IDs de productos similares
        similar_product_ids = []
        for _, row in product_pairs.iterrows():
            if row['producto1_id'] == product_id:
                similar_product_ids.append(row['producto2_id'])
            else:
                similar_product_ids.append(row['producto1_id'])
        
        if not similar_product_ids:
            return pd.DataFrame()
            
        # Obtener detalles de productos
        similar_products = self.data_manager.get_product_details(similar_product_ids, tenant_id)
        
        if similar_products.empty:
            return pd.DataFrame()
        
        # Agregar puntuación de similitud
        similar_products['score'] = 0.0
        for i, pid in enumerate(similar_product_ids):
            idx = similar_products.index[similar_products['producto_id'] == pid].tolist()
            if idx:
                # Puntuar según la posición en la lista (más alto = más similar)
                similar_products.loc[idx[0], 'score'] = 1.0 - (i * 0.1)
        
        # Guardar en caché
        self.product_similarities[cache_key] = similar_products
        
        return similar_products
    
    def _add_recommendation_reasons(self, recommendations, purchase_history, context=None):
        """Añade razones a las recomendaciones basadas en el contexto y preferencias"""
        if recommendations.empty:
            return
            
        # Preparar razones predeterminadas por categoría
        category_reasons = {}
        if not purchase_history.empty:
            # Verificar si existe la columna categoria_id
            if 'categoria_id' in purchase_history.columns:
                # Filtrar nulos y obtener las más frecuentes
                valid_categories = purchase_history.dropna(subset=['categoria_id'])
                if not valid_categories.empty:
                    preferred_categories = valid_categories['categoria_id'].value_counts().head(2).index.tolist()
                    for i, cat_id in enumerate(preferred_categories):
                        if i == 0:
                            category_reasons[cat_id] = "Basado en tus platos favoritos"
                        else:
                            category_reasons[cat_id] = "Similar a productos que has disfrutado"
        
        # Añadir razones contextuales
        time_reasons = {
            'MORNING': 'Ideal para el desayuno',
            'LUNCH': 'Perfecto para el almuerzo',
            'AFTERNOON': 'Excelente para la tarde',
            'DINNER': 'Recomendado para la cena'
        }
        
        # Razones basadas en tamaño de grupo
        party_reasons = {
            1: "Ideal para disfrutar solo",
            2: "Perfecto para compartir en pareja",
            3: "Ideal para un grupo pequeño",
            4: "Buena opción para grupos"
        }
        
        # Asignar razón predeterminada
        recommendations['reason'] = "Recomendado para ti"
        
        # Asignar razones por categoría si la columna existe
        if 'categoria_id' in recommendations.columns:
            for cat_id, reason in category_reasons.items():
                # Manejar posibles valores nulos
                mask = recommendations['categoria_id'].fillna('') == cat_id
                if mask.any():
                    recommendations.loc[mask, 'reason'] = reason
            
        # Asignar razones basadas en contexto
        if context:
            if 'timeOfDay' in context and context['timeOfDay'] in time_reasons:
                # Aplicar razón por hora a algunos productos aleatoriamente
                if len(recommendations) > 3:
                    try:
                        random_indices = np.random.choice(recommendations.index, size=min(2, len(recommendations)), replace=False)
                        recommendations.loc[random_indices, 'reason'] = time_reasons[context['timeOfDay']]
                    except Exception as e:
                        self.logger.warning(f"Error asignando razones por hora: {str(e)}")
                    
            if 'partySize' in context and context['partySize'] in party_reasons:
                # Aplicar razón por tamaño de grupo a ciertos productos
                if len(recommendations) > 3:
                    try:
                        random_indices = np.random.choice(recommendations.index, size=min(2, len(recommendations)), replace=False)
                        recommendations.loc[random_indices, 'reason'] = party_reasons[context['partySize']]
                    except Exception as e:
                        self.logger.warning(f"Error asignando razones por tamaño de grupo: {str(e)}")
    
    def _format_recommendations(self, recommendations, tenant_id):
        """Formatea las recomendaciones en el formato esperado por GraphQL"""
        if recommendations.empty:
            return []
        
        # Identificar columnas con los datos correctos
        id_col = None
        name_col = None
        price_col = None
        desc_col = None
        cat_id_col = None
        cat_name_col = None
        
        # Buscar columnas de producto_id
        for col in recommendations.columns:
            if 'producto_id' == col:
                id_col = col
                break
            elif 'producto_id' in col:
                id_col = col
                
        # Buscar columnas de nombre
        for col in recommendations.columns:
            if 'nombre' == col and recommendations[col].notna().any():
                name_col = col
                break
            elif 'nombre' in col and recommendations[col].notna().any():
                name_col = col
                
        # Buscar columnas de precio
        for col in recommendations.columns:
            if 'precio' == col and recommendations[col].notna().any():
                price_col = col
                break
            elif 'precio' in col and recommendations[col].notna().any():
                price_col = col
                
        # Buscar columnas de descripción
        for col in recommendations.columns:
            if 'descripcion' == col and recommendations[col].notna().any():
                desc_col = col
                break
            elif 'descripcion' in col and recommendations[col].notna().any():
                desc_col = col
                
        # Buscar columnas de categoria_id
        for col in recommendations.columns:
            if 'categoria_id' == col and recommendations[col].notna().any():
                cat_id_col = col
                break
            elif 'categoria_id' in col and recommendations[col].notna().any():
                cat_id_col = col
                
        # Buscar columnas de categoria_nombre
        for col in recommendations.columns:
            if 'categoria_nombre' == col and recommendations[col].notna().any():
                cat_name_col = col
                break
            elif 'categoria_nombre' in col and recommendations[col].notna().any():
                cat_name_col = col
        
        # Si no encontramos columnas esenciales, usar los nombres de columnas más probables
        if id_col is None:
            self.logger.warning("No se encontró columna de producto_id, usando primera columna")
            id_col = recommendations.columns[0]
        
        if name_col is None:
            self.logger.warning("No se encontró columna de nombre, usando valor por defecto")
            name_col = id_col  # Usaremos el ID como nombre en último caso
            
        result = []
        for _, row in recommendations.iterrows():
            try:
                # Extraer valores usando las columnas identificadas
                producto_id = row[id_col] if id_col else ""
                nombre = row[name_col] if name_col and pd.notna(row[name_col]) else f"Producto {producto_id}"
                precio = float(row[price_col]) if price_col and pd.notna(row[price_col]) else 0.0
                descripcion = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else ""
                
                # Categoría
                cat_id = str(row[cat_id_col]) if cat_id_col and pd.notna(row[cat_id_col]) else ""
                cat_name = str(row[cat_name_col]) if cat_name_col and pd.notna(row[cat_name_col]) else "Desconocida"
                
                # Construir objeto producto
                product = {
                    "id": str(producto_id),
                    "name": str(nombre),
                    "description": descripcion,
                    "price": precio,
                    "category": {
                        "id": cat_id,
                        "name": cat_name
                    },
                    "imageUrl": f"/images/{str(nombre).lower().replace(' ', '-').replace(',', '')}.jpg"
                }
                
                # Determinar la razón y puntuación
                reason = str(row.get('reason', "Recomendado para ti"))
                
                # Determinar la puntuación, buscando diferentes posibles columnas
                score = 0.5  # Valor predeterminado
                for score_col in ['final_score', 'score', 'pop_score']:
                    if score_col in row and pd.notna(row[score_col]):
                        score = float(row[score_col])
                        break
                
                # Construir la recomendación
                result.append({
                    "product": product,
                    "score": score,
                    "reason": reason
                })
            except Exception as e:
                self.logger.error(f"Error al formatear recomendación: {str(e)}")
                continue
                
        return result