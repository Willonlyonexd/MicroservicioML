from db.mongo_client import get_mongo_manager

def migrate_collections():
    mongo = get_mongo_manager()
    
    collections = [
        "predicciones_productos",
        "predicciones_ventas",
        "ml_predicciones",
        "ml_predicciones_categoria"
    ]
    
    for collection in collections:
        if collection in mongo.db.list_collection_names():
            print(f"Actualizando colección: {collection}")
            result = mongo.db[collection].update_many(
                {"tenant_id": {"$exists": False}},
                {"$set": {"tenant_id": 1}}
            )
            print(f"- {result.modified_count} documentos actualizados")
        else:
            print(f"La colección {collection} no existe - se creará cuando ejecutes el forecasting")

    # Crear directorios
    import os
    tenant_id = 1
    directories = [
        f"models/forecast/tenant_{tenant_id}",
        f"models/forecast/tenant_{tenant_id}/products",
        f"plots/tenant_{tenant_id}"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directorio creado: {directory}")

if __name__ == "__main__":
    print("Iniciando migración para soporte multi-tenant...")
    migrate_collections()
    print("✅ Migración completada")