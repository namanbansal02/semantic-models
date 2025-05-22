from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

app = FastAPI(title="Semantic Similarity API")

MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"
MODEL_CACHE_PATH = "model_cache.joblib"

# Load or cache the model
def load_model():
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            model = joblib.load(MODEL_CACHE_PATH)
            print("Model loaded from cache.")
            return model
        except Exception as e:
            print(f"Failed to load cached model: {e}")
    model = SentenceTransformer(MODEL_NAME)
    joblib.dump(model, MODEL_CACHE_PATH)
    print("Model downloaded and cached.")
    return model

model = load_model()

# Request & Response schemas
class TextPair(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity_score: float
    class Config:
        allow_population_by_field_name = True
        fields = {"similarity_score": {"alias": "similarity score"}}
        populate_by_name = True
        json_encoders = {
            float: lambda v: round(v, 4)
        }


# Utility: compute similarity
def compute_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode(text1, convert_to_numpy=True,normalize_embeddings=True, show_progress_bar=False, device='cpu',truncation=True)
    emb2 = model.encode(text2, convert_to_numpy=True,normalize_embeddings=True, show_progress_bar=False, device='cpu',truncation=True)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return round((score + 1) / 2, 4)  # Normalize [-1, 1] to [0, 1]

# POST endpoint
@app.post("/similarity", response_model=SimilarityResponse,response_model_by_alias=True)
async def similarity_endpoint(data: TextPair):
    if not data.text1.strip() or not data.text2.strip():
        raise HTTPException(status_code=400, detail="Both 'text1' and 'text2' must be non-empty.")
    
    score = compute_similarity(data.text1, data.text2)
    return SimilarityResponse(similarity_score=score)

# Health check
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Semantic Similarity API is up. POST /similarity"}
