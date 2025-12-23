from qdrant_client import QdrantClient
client = QdrantClient(location=":memory:")
print("Methods:", dir(client))
try:
    client.search
    print("search exists")
except AttributeError:
    print("search MISSING")
