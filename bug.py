from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client["microservicio_ml"]  # Ajusta si tu base se llama diferente

# Rango del 2 de junio 2025
start = datetime(2025, 6, 2, 0, 0, 0)
end = datetime(2025, 6, 3, 0, 0, 0)

# 1. Obtener cuenta_mesa_id de ventas del 2 de junio
ventas = db.raw_venta.find({"fecha_venta": {"$gte": start, "$lt": end}}, {"cuenta_mesa_id": 1})
cuentas_afectadas = [v["cuenta_mesa_id"] for v in ventas]

# 2. Obtener pedido_id de esas cuentas
pedidos = db.raw_pedido.find({"cuenta_mesa_id": {"$in": cuentas_afectadas}}, {"pedido_id": 1})
pedidos_afectados = [p["pedido_id"] for p in pedidos]

# 3. Eliminar detalles de pedidos
result_detalle = db.raw_pedido_detalle.delete_many({"pedido_id": {"$in": pedidos_afectados}})
print(f"Detalles eliminados: {result_detalle.deleted_count}")

# 4. Eliminar pedidos
result_pedido = db.raw_pedido.delete_many({"pedido_id": {"$in": pedidos_afectados}})
print(f"Pedidos eliminados: {result_pedido.deleted_count}")

# 5. Eliminar ventas
result_venta = db.raw_venta.delete_many({"fecha_venta": {"$gte": start, "$lt": end}})
print(f"Ventas eliminadas: {result_venta.deleted_count}")

# 6. Eliminar cuentas de mesa asociadas
result_cuentas = db.raw_cuenta_mesa.delete_many({"id": {"$in": cuentas_afectadas}})
print(f"Cuentas de mesa eliminadas: {result_cuentas.deleted_count}")
