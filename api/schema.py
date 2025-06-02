# Schema GraphQL para el sistema de predicción de ventas con contexto multi-tenant
schema_sdl = """
type Query {
  # Predicciones diarias
  dailyForecasts(days: Int = 7): [DailyPrediction!]!
  
  # Predicciones semanales
  weeklyForecasts(weeks: Int = 3): [WeeklyPrediction!]!
  
  # Predicciones mensuales
  monthlyForecasts(months: Int = 3): [MonthlyPrediction!]!
  
  # Predicción para un producto específico
  productForecast(productId: Int!, days: Int = 7): ProductForecast
  
  # Predicción para una categoría específica
  categoryForecast(categoryId: Int!, days: Int = 7): CategoryForecast
  
  # Estado de los modelos
  modelStatus: ModelStatus!
  
  # Visualizaciones disponibles
  availableVisualizations: AvailableVisualizations!
  
  # NUEVAS CONSULTAS PARA DATOS COMBINADOS (HISTÓRICO + PREDICCIÓN)
  
  # Datos históricos y predicciones diarias combinados
  salesWithForecast(history_days: Int = 30, forecast_days: Int = 7): CombinedSalesData!
  
  # Datos históricos y predicciones semanales combinados
  weeklyWithForecast(history_weeks: Int = 12, forecast_weeks: Int = 3): CombinedWeeklyData!
  
  # Datos históricos y predicciones mensuales combinados
  monthlyWithForecast(history_months: Int = 12, forecast_months: Int = 3): CombinedMonthlyData!
  
  # Datos históricos y predicciones para un producto específico
  productHistoricalAndForecast(productId: Int!, history_days: Int = 30, forecast_days: Int = 7): CombinedProductData
  
  # Datos históricos y predicciones para una categoría específica
  categoryHistoricalAndForecast(categoryId: Int!, history_days: Int = 30, forecast_days: Int = 7): CombinedCategoryData
  
  # CONSULTAS PARA SEGMENTACIÓN DE CLIENTES
  
  # Obtener todos los clientes segmentados, con filtro opcional por cluster
  clientesSegmentados(clusterId: Int): [ClienteSegmento!]!
  
  # Obtener información detallada de un cluster específico
  clusterInfo(clusterId: Int!): ClusterInfo
  
  # Obtener información de todos los clusters
  todosClusters: [ClusterInfo!]!
  
  # NUEVAS CONSULTAS ESPECÍFICAS PARA GRÁFICOS
  
  # Distribución general de segmentos para gráfico de pastel
  segmentacionGeneral: SegmentacionGeneral!
  
  # Lista detallada de clientes con su información de segmentación
  clientesConSegmentacion: [ClienteConSegmento!]!
}

# Predicción diaria de ventas
type DailyPrediction {
  fecha: String!
  prediccion: Float!
  confianza: Float!
  timestamp: String
  generado_en: String
  generado_por: String
  tenant_id: Int
}

# Predicción semanal de ventas
type WeeklyPrediction {
  periodo: String!
  fecha_inicio: String!
  fecha_fin: String!
  prediccion: Float!
  confianza: Float!
  timestamp: String
  tenant_id: Int
}

# Predicción mensual de ventas
type MonthlyPrediction {
  periodo: String!
  fecha_inicio: String!
  fecha_fin: String!
  prediccion: Float!
  confianza: Float!
  timestamp: String
  tenant_id: Int
}

# Predicción de un producto específico
type ProductForecast {
  producto_id: Int!
  nombre: String!
  categoria_id: Int
  categoria: String
  predicciones: [ProductPrediction!]!
  tenant_id: Int
}

# Predicción diaria para un producto
type ProductPrediction {
  fecha: String!
  prediccion: Float!
  confianza: Float!
}

# Predicción de una categoría
type CategoryForecast {
  categoria_id: Int!
  nombre_categoria: String!
  predicciones: [CategoryPrediction!]!
  tenant_id: Int
}

# Predicción diaria para una categoría
type CategoryPrediction {
  fecha: String!
  prediccion: Float!
  confianza: Float!
  productos: [Int!]!
}

# Estado de los modelos de ML
type ModelStatus {
  general: GeneralModelStatus!
  productos: [ProductModelStatus!]!
}

# Estado del modelo general
type GeneralModelStatus {
  entrenado: Boolean!
  ultima_actualizacion: String
  exactitud: Float
  error_mae: Float
  tenant_id: Int
}

# Estado del modelo de un producto
type ProductModelStatus {
  producto_id: Int!
  nombre: String!
  entrenado: Boolean!
  ultima_actualizacion: String
  exactitud: Float
  tenant_id: Int
}

# Visualizaciones disponibles
type AvailableVisualizations {
  general: GeneralVisualization
  productos: [ProductVisualization!]!
  categorias: [CategoryVisualization!]!
}

# Visualización general
type GeneralVisualization {
  url: String!
  fecha_generacion: String!
  tenant_id: Int
}

# Visualización para un producto
type ProductVisualization {
  producto_id: Int!
  nombre: String!
  url: String!
  fecha_generacion: String!
  tenant_id: Int
}

# Visualización para una categoría
type CategoryVisualization {
  categoria_id: Int!
  nombre: String!
  url: String!
  fecha_generacion: String!
  tenant_id: Int
}

# NUEVOS TIPOS PARA DATOS COMBINADOS (HISTÓRICO + PREDICCIÓN)

# Dato histórico de ventas
type HistoricalDataPoint {
  fecha: String!
  ventas: Float!
  tipo: String!
  tenant_id: Int
}

# Datos históricos semanales o mensuales
type HistoricalAggregatedData {
  periodo: String!
  fecha_inicio: String!
  fecha_fin: String!
  ventas: Float!
  tipo: String!
  tenant_id: Int
}

# Datos históricos y predicciones diarias combinados
type CombinedSalesData {
  current_date: String!
  historical: [HistoricalDataPoint!]!
  forecast: [DailyPrediction!]!
  tenant_id: Int
}

# Datos históricos y predicciones semanales combinados
type CombinedWeeklyData {
  current_date: String!
  historical: [HistoricalAggregatedData!]!
  forecast: [WeeklyPrediction!]!
  tenant_id: Int
}

# Datos históricos y predicciones mensuales combinados
type CombinedMonthlyData {
  current_date: String!
  historical: [HistoricalAggregatedData!]!
  forecast: [MonthlyPrediction!]!
  tenant_id: Int
}

# Datos históricos y predicciones para un producto específico
type CombinedProductData {
  producto_id: Int!
  nombre_producto: String!
  categoria_id: Int
  current_date: String!
  historical: [HistoricalDataPoint!]!
  forecast: [ProductPrediction!]!
  tenant_id: Int
}

# Datos históricos y predicciones para una categoría específica
type CombinedCategoryData {
  categoria_id: Int!
  nombre_categoria: String!
  current_date: String!
  historical: [HistoricalDataPoint!]!
  forecast: [CategoryPrediction!]!
  tenant_id: Int
}

# TIPOS PARA SEGMENTACIÓN DE CLIENTES

# Cliente segmentado
type ClienteSegmento {
  clienteId: ID!
  cluster: Int!
  clusterNombre: String!
  numVisitas: Int!
  comensalesPromedio: Float!
  gastoTotal: Float!
  tenantId: Int
}

# Información de un cluster
type ClusterInfo {
  clusterId: Int!
  nombre: String!
  numClientes: Int!
  gastoPromedio: Float!
  visitasPromedio: Float!
  comensalesPromedio: Float!
  descripcion: String!
  tenantId: Int
}

# NUEVOS TIPOS PARA VISUALIZACIONES ESPECÍFICAS

# Distribución general de segmentos
type SegmentacionGeneral {
  total: Int!
  distribucion: [SegmentoCount!]!
  tenantId: Int
}

# Conteo por segmento para gráfico de pastel
type SegmentoCount {
  nombre: String!
  cantidad: Int!
  porcentaje: Float!
}

# Cliente con información detallada de segmentación
type ClienteConSegmento {
  clienteId: ID!
  nombreCompleto: String!
  segmento: String!
  numVisitas: Int!
  comensalesPromedio: Float!
  gastoTotal: Float!
  tenantId: Int
}
"""