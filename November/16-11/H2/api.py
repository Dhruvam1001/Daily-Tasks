import pandas as pd
import time
import math
import joblib
from collections import OrderedDict
from threading import Lock
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()
bundle = joblib.load(Path(__file__).with_name("best_model_v1.joblib"))
model = bundle["model"]
features = bundle["features"]
cache = OrderedDict()
lock = Lock()
cache_hits = 0
cache_misses = 0

@app.middleware("http")
async def measure_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = str(round((time.perf_counter() - start) * 1000, 2))
    return response

@app.get("/")
def home():
    return {"message": "Heart Disease API is running"}

@app.post("/predict")
def predict(inputs: dict[str, float]):
    global cache_hits, cache_misses

    missing = [c for c in features if c not in inputs]
    extra = [c for c in inputs if c not in features]

    if missing or extra:
        raise HTTPException(400, detail="Incorrect input features")

    values = tuple(float(inputs[c]) for c in features)

    if not all(math.isfinite(v) for v in values):
        raise HTTPException(422, detail="Invalid feature values")

    with lock:
        if values in cache:
            cache_hits += 1
            result = cache[values]
            cache.move_to_end(values)

            return {
                **result,
                "data_processing_ms": 0,
                "inference_ms": 0,
                "cache_hit": True
            }

        cache_misses += 1

    start = time.perf_counter()
    row = pd.DataFrame([values], columns=features)
    data_time = (time.perf_counter() - start) * 1000
    start = time.perf_counter()

    if callable(getattr(model, "predict_proba", None)):
        probabilities = model.predict_proba(row)[0]
        index = probabilities.argmax()
        prediction = int(model.classes_[index])
        confidence = float(probabilities[index] * 100)
    else:
        prediction = int(model.predict(row)[0])
        confidence = None

    inference_time = (time.perf_counter() - start) * 1000

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 2) if confidence is not None else None
    }

    with lock:
        cache[values] = result
        if len(cache) > 5000:
            cache.popitem(last=False)

    return {
        **result,
        "data_processing_ms": round(data_time, 2),
        "inference_ms": round(inference_time, 2),
        "cache_hit": False
    }

@app.get("/performance")
def performance():
    with lock:
        total = cache_hits + cache_misses
        return {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_entries": len(cache),
            "cache_hit_rate": round(cache_hits / total * 100, 2)
            if total else 0
        }