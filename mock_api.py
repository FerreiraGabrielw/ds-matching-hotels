#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn, random

app = FastAPI(title="Mock Hotel Enrichment API", version="1.0")

AMENITIES = ["wifi","pool","gym","spa","bar","restaurant","parking","air_conditioning","pet_friendly"]

class EnrichRequest(BaseModel):
    hotel_name: str
    city: str
    country: str

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.post("/enrich")
def enrich(req: EnrichRequest):
    # deterministic-ish enrichment based on hashing
    key = (req.hotel_name + req.city + req.country).lower()
    rng = random.Random(hash(key) % (2**32 - 1))
    stars = rng.choice([3,4,5])
    score = round(rng.uniform(7.0, 9.7), 1)
    amns = rng.sample(AMENITIES, k=rng.randint(3,6))
    return {"category_stars": stars, "review_score": score, "amenities": amns}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
