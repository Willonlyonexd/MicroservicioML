from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from ariadne import make_executable_schema, graphql_sync
from ariadne import ObjectType, QueryType
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar nuestro esquema y resolvers
from api.schema import schema_sdl
from api.resolvers import resolvers

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
    
    # Log de la consulta recibida (nivel debug)
    logger.debug(f"GraphQL Query: {data}")
    
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
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
    logger.info(f"Sirviendo gráfico: {filename}")
    return send_from_directory('plots', filename)

# Ruta para health check
@app.route('/health')
def health_check():
    logger.info("Health check solicitado")
    return jsonify({
        "status": "ok", 
        "service": "restaurant-ml-forecast-api",
        "version": "1.0.0",
        "timestamp": "2025-05-31 02:53:03",
        "user": "muimui69"
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
    app.run(debug=True, host='0.0.0.0', port=5000)