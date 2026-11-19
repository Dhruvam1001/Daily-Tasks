import os
import random
import pandas as pd
from pathlib import Path
from locust import HttpUser, task, between

file = Path(__file__).resolve().parents[1] / "heart.csv"
data = pd.read_csv(file).drop(columns=["target"])

PAYLOADS = [{c: float(v) for c, v in row.items()} for row in data.to_dict(orient="records")]

TOKEN = os.getenv("API_TOKEN", "")

class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def predict(self):
        payload = random.choice(PAYLOADS)
        headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

        with self.client.post("/predict", json = payload, headers = headers, name = "/predict", catch_response = True) as response:

            if response.status_code != 200: 
                response.failure("Prediction failed")

            else:
                result = response.json()
                server_time = response.headers.get("X-Response-Time-ms")

                if server_time:
                    self.environment.events.request.fire(request_type = "SERVER", name = "API Processing", response_time = float(server_time), response_length = 0, response = response, context = {}, exception = None)

                if result.get("inference_ms", 0) > 0:
                    self.environment.events.request.fire(request_type = "MODEL", name = "Model Inference", response_time = result["inference_ms"], response_length = 0, response = response, context = {}, exception = None)