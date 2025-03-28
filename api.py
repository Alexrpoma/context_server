from fastapi import FastAPI

from llm_service import LLMService
from models import DataItem

app = FastAPI(title="Context Server - HTTP Receiver")
llm_service = LLMService()

@app.post("/context-llm-api/process-data", status_code=202)
async def process_data_endpoint(data_item: DataItem):
    print(f"LLM Service: Received {data_item.url} items to process.")
    try:
        llm_service.run(data_item.summary, 512)
    except Exception as e:
        print(f"Error process {data_item.url}: {e}")
    return {"message": "Data received and processing started."}