# Define el esquema SDL para la segmentación de clientes
segmentation_schema_sdl = """
# Tipo para representar un cliente segmentado
type ClienteSegmento {
  clienteId: ID!
  cluster: Int!
  clusterNombre: String!
  numVisitas: Int!
  comensalesPromedio: Float!
  gastoTotal: Float!
  tenantId: Int
}

# Tipo para representar la información de un cluster
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

extend type Query {
  # Obtener todos los clientes segmentados, con filtro opcional por cluster
  clientesSegmentados(clusterId: Int): [ClienteSegmento!]!
  
  # Obtener información detallada de un cluster específico
  clusterInfo(clusterId: Int!): ClusterInfo
  
  # Obtener información de todos los clusters
  todosClusters: [ClusterInfo!]!
}
"""