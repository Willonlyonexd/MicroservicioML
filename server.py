# Este archivo simplemente re-exporta el servidor desde api/server.py
from api.server import app

# Si necesitas alguna configuración adicional para el servidor, puedes hacerla aquí
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)