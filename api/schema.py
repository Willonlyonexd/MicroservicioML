# Schema GraphQL en SDL (Schema Definition Language)
schema_sdl = """
type DailyPrediction {
  fecha: String!
  prediccion: Float!
  confianza: Float!
  timestamp: String!
  generado_en: String
  generado_por: String
}

type AggregatedPrediction {
  periodo: String!
  fecha_inicio: String!
  fecha_fin: String!
  prediccion: Float!
  confianza: Float!
  timestamp: String!
}

type ProductPrediction {
  fecha: String!
  prediccion: Float!
  confianza: Float!
}

type ProductForecast {
  producto_id: Int!
  nombre: String!
  categoria_id: Int!
  categoria: String!
  predicciones: [ProductPrediction!]!
}

type CategoryPrediction {
  fecha: String!
  prediccion: Float!
  confianza: Float!
  productos: [Int!]!
}

type CategoryForecast {
  categoria_id: Int!
  nombre_categoria: String!
  predicciones: [CategoryPrediction!]!
}

type Visualization {
  url: String!
  fecha_generacion: String!
}

type ProductVisualization {
  producto_id: Int!
  nombre: String!
  url: String!
  fecha_generacion: String!
}

type CategoryVisualization {
  categoria_id: Int!
  nombre: String!
  url: String!
  fecha_generacion: String!
}

type AvailableVisualizations {
  general: Visualization
  productos: [ProductVisualization!]!
  categorias: [CategoryVisualization!]!
}

type ModelInfo {
  entrenado: Boolean!
  ultima_actualizacion: String
  exactitud: Float
  error_mae: Float
}

type ProductModelInfo {
  producto_id: Int!
  nombre: String!
  entrenado: Boolean!
  ultima_actualizacion: String
  exactitud: Float
}

type ModelStatus {
  general: ModelInfo!
  productos: [ProductModelInfo!]!
}

type Query {
  dailyForecasts(days: Int = 7): [DailyPrediction!]!
  weeklyForecasts(weeks: Int = 1): [AggregatedPrediction!]!
  monthlyForecasts(months: Int = 1): [AggregatedPrediction!]!
  productForecast(productId: Int!, days: Int = 7): ProductForecast
  categoryForecast(categoryId: Int!, days: Int = 7): CategoryForecast
  modelStatus: ModelStatus!
  availableVisualizations: AvailableVisualizations!
}
"""