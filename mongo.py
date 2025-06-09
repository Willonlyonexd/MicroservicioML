
from pymongo import MongoClient

# Conexión a MongoDB local
cliente_local = MongoClient("mongodb://localhost:27017")
db_local = cliente_local["microservicio_ml"]
colecciones = db_local.list_collection_names()

# Conexión a MongoDB Atlas
cliente_atlas = MongoClient("mongodb+srv://houwenvt:will@cluster0.crz8eun.mongodb.net/")
db_atlas = cliente_atlas["microservicio_ml"]

print(f"Migrando colecciones: {colecciones}\n")

# Migrar cada colección
for nombre_col in colecciones:
    datos = list(db_local[nombre_col].find())
    if datos:
        db_atlas[nombre_col].insert_many(datos)
        print(f"✅ {nombre_col}: {len(datos)} documentos copiados")
    else:
        print(f"⚠️ {nombre_col} está vacía o no se pudo copiar")

print("\n🚀 Migración completada.")

