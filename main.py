from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()

# Încarcă datele
df = pd.read_excel("produse6.xlsx")
df["full_text"] = df["Descriere Produs"].fillna("").astype(str)

# Încarcă modelul și creează indexul
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

product_metadata = df[[
    "Denumire Produs", "Descriere Produs", "Pret",
    "Imagine principala", "Categorie / Categorii",
    "Link Canonic Implicit Produs"
]]

# Model pentru request
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
