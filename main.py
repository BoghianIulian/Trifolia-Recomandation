from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Încarcă embeddings preprocesați și metadatele
embeddings = np.load("embeddings.npy")
df = pd.read_pickle("metadata.pkl")

# Creează index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Selectează doar coloanele relevante
product_metadata = df[[
    "Denumire Produs", "Descriere Produs", "Pret",
    "Imagine principala", "Categorie / Categorii",
    "Link Canonic Implicit Produs"
]]

# Încarcă modelul (doar o dată)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Modelul cererii primite
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_products(request: QueryRequest):
    query_embedding = model.encode([request.query])
    distances, indices = index.search(np.array(query_embedding), request.top_k)
    results = []
    for i in indices[0]:
        produs = product_metadata.iloc[i]
        results.append({
            "denumire": produs.get("Denumire Produs", "Fără titlu"),
            "pret": str(produs.get("Pret", "N/A")),
            "descriere": str(produs.get("Descriere Produs", ""))[:200],
            "link": produs.get("Link Canonic Implicit Produs", "")
        })
    return {"rezultate": results}
