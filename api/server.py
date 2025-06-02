from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from ariadne import make_executable_schema, graphql_sync
from ariadne import ObjectType, QueryType
import logging
from datetime import datetime
import os
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar el path para poder importar desde la carpeta api
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Si estamos ejecutando directamente este archivo desde la carpeta api/
if current_dir.endswith('api'):
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Importaciones relativas cuando ejecutamos desde api/
    try:
        from schema import schema_sdl
        from resolvers import resolvers
        import segmentation_init
        init_segmentation = segmentation_init.init_segmentation
        logger.info("Importaciones relativas exitosas")
    except ImportError:
        # Si fallan las importaciones relativas, intentar con importaciones absolutas
        from api.schema import schema_sdl
        from api.resolvers import resolvers
        from api.segmentation_init import init_segmentation
        logger.info("Importaciones absolutas exitosas")
else:
    # Importaciones absolutas cuando ejecutamos desde la raíz
    from api.schema import schema_sdl
    from api.resolvers import resolvers
    from api.segmentation_init import init_segmentation
    logger.info("Importaciones absolutas desde raíz exitosas")

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Configurar los tipos para Ariadne
query = QueryType()

# Registrar resolvers en el tipo Query
for field, resolver in resolvers["Query"].items():
    query.set_field(field, resolver)

# Crear el esquema ejecutable
schema = make_executable_schema(schema_sdl, query)

# Inicializar módulo de segmentación
init_segmentation()

# El resto del archivo sigue igual...
# Definir HTML del playground manualmente (ya que PLAYGROUND_HTML no está disponible)
PLAYGROUND_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GraphQL Playground</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="user-scalable=no, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, minimal-ui" />
    <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/graphql-playground-react@1.7.22/build/static/css/index.css" />
    <link rel="shortcut icon" href="//cdn.jsdelivr.net/npm/graphql-playground-react@1.7.22/build/favicon.png" />
    <script src="//cdn.jsdelivr.net/npm/graphql-playground-react@1.7.22/build/static/js/middleware.js"></script>
</head>
<body>
    <div id="root"></div>
    <script>
        window.addEventListener('load', function(event) {
            GraphQLPlayground.init(document.getElementById('root'), {
                endpoint: '/graphql',
                settings: {
                    'request.credentials': 'include',
                    'tracing.hideTracingResponse': true,
                    'editor.theme': 'dark',
                    'editor.fontFamily': "'Source Code Pro', 'Consolas', 'Inconsolata', 'Droid Sans Mono', 'Monaco', monospace",
                    'editor.fontSize': 14,
                },
                headers: {
                    'X-Tenant-ID': '1'  // Valor predeterminado para el playground
                }
            })
        })
    </script>
</body>
</html>
"""

# Ruta para GraphQL
@app.route("/graphql", methods=["GET"])
def graphql_playground():
    """Playground de GraphQL para pruebas interactivas"""
    return PLAYGROUND_HTML, 200

@app.route("/graphql", methods=["POST"])
def graphql_server():
    """Endpoint para consultas GraphQL"""
    data = request.get_json()
    
    # Extraer tenant_id del encabezado HTTP
    tenant_id = request.headers.get('X-Tenant-ID', '1')
    
    # Convertir a entero si es posible
    try:
        tenant_id = int(tenant_id)
    except (ValueError, TypeError):
        tenant_id = 1  # Valor predeterminado si no es un número válido
    
    # Crear objeto de contexto
    context = {
        "request": request,
        "tenant_id": tenant_id,
        "headers": dict(request.headers)
    }
    
    # Log de la consulta recibida
    logger.info(f"GraphQL Query para tenant {tenant_id}")
    logger.debug(f"Query details: {data}")
    
    success, result = graphql_sync(
        schema,
        data,
        context_value=context,
        debug=app.debug
    )
    
    # Log de errores si los hay
    if not success:
        logger.error(f"GraphQL Error: {result}")
    
    status_code = 200 if success else 400
    return jsonify(result), status_code

# Ruta para servir los archivos estáticos (gráficos)
@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    # Extraer tenant_id del encabezado HTTP
    tenant_id = request.headers.get('X-Tenant-ID', '1')
    try:
        tenant_id = int(tenant_id)
    except (ValueError, TypeError):
        tenant_id = 1
        
    logger.info(f"Sirviendo gráfico: {filename} para tenant {tenant_id}")
    
    # Primero intentar servir desde el directorio específico del tenant
    tenant_path = f'plots/tenant_{tenant_id}'
    try:
        return send_from_directory(tenant_path, filename)
    except:
        # Si no existe, servir desde el directorio general
        return send_from_directory('plots', filename)

# Nueva ruta para servir gráficos de segmentación
@app.route('/static/segmentation/<path:filename>')
def serve_segmentation_plot(filename):
    # Extraer tenant_id del encabezado HTTP
    tenant_id = request.headers.get('X-Tenant-ID', '1')
    try:
        tenant_id = int(tenant_id)
    except (ValueError, TypeError):
        tenant_id = 1
        
    logger.info(f"Sirviendo gráfico de segmentación: {filename} para tenant {tenant_id}")
    
    # Primero intentar servir desde el directorio específico del tenant
    tenant_path = f'plots/tenant_{tenant_id}/segmentation'
    try:
        return send_from_directory(tenant_path, filename)
    except:
        # Si no existe, intentar servir desde un directorio general de segmentación
        try:
            return send_from_directory('plots/segmentation', filename)
        except:
            # Si tampoco existe, servir desde el directorio general
            return send_from_directory('plots', filename)

# Ruta para health check
@app.route('/health')
def health_check():
    logger.info("Health check solicitado")
    # Usar fecha actual para el timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "status": "ok", 
        "service": "restaurant-ml-forecast-api",
        "version": "1.2.0",  # Actualizado a 1.2.0 para indicar soporte de segmentación
        "timestamp": current_time,
        "user": "muimui69",
        "features": {
            "combined_visualization": True,  # Indica soporte para visualización combinada
            "current_date_pivot": "2025-05-31",  # Fecha actual como punto pivote
            "customer_segmentation": True  # Nueva característica: segmentación de clientes
        }
    })

# Manejador de errores para rutas no encontradas
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Ruta no encontrada"}), 404

# Manejador de errores para excepciones internas
@app.errorhandler(500)
def server_error(e):
    logger.error(f"Error interno del servidor: {str(e)}")
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    logger.info("Iniciando servidor GraphQL con Ariadne...")
    logger.info("Playground disponible en: http://localhost:5000/graphql")
    logger.info("Soporte multi-tenant activado (usar encabezado HTTP 'X-Tenant-ID')")
    logger.info("Visualización combinada disponible (histórico + predicción con fecha pivote 2025-05-31)")
    logger.info("Segmentación de clientes disponible (K-means en 4 clusters: VIP, PREMIUM, REGULAR, OCASIONAL)")
    app.run(debug=True, host='0.0.0.0', port=5000)