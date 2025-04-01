import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
# from fastapi.responses import HTMLResponse

from connection_manager import ConnectionManager
from llm_service import LLMService
from log_config.logging_config import setup_logging
from models import DataItem
import asyncio

setup_logging()
app = FastAPI(title="Context Server - HTTP Receiver",
              description="API for receiving data and streaming LLM responses via WebSocket.",
              version="1.0.0")
llm_service = LLMService()
manager = ConnectionManager()

@app.get("/")
async def get():
    return {"message": "Context Server is running."}

# --- Async Streaming Consumer ---
async def stream_tokens_to_client(client_id: str, queue: asyncio.Queue):
    """Consumes from the queue and sends tokens to the specific WebSocket client."""
    logging.info(f"Starting token streaming consumer for {client_id}...")
    try:
        while True:
            token_or_signal = await queue.get()
            if token_or_signal is None:
                logging.info(f"End signal received internally for {client_id}. Sending [DONE] signal.")
                try:
                    await manager.send_personal_message("[DONE]", client_id)
                except Exception as send_error:
                    logging.error(f"Error sending [DONE] signal to {client_id}: {send_error}")
                break
            # print(f"Sending token to {client_id}: {token_or_signal[:30]}...") # Debug
            await manager.send_personal_message(token_or_signal, client_id)
            queue.task_done()
    except asyncio.CancelledError:
         logging.error(f"Streaming task for {client_id} was cancelled.")
    except Exception as e:
         logging.error(f"Error in streaming consumer for {client_id}: {e}")
         await manager.send_personal_message(f"ERROR: Streaming failed - {e}", client_id)
    finally:
         logging.info(f"Token streaming consumer for {client_id} finished.")
         manager.remove_streaming_task(client_id)

@app.post("/context-llm-api/process-data", status_code=202)
async def process_data_endpoint(request_item: DataItem, background_tasks: BackgroundTasks):
    client_id = request_item.client_id
    summary = request_item.summary
    url = request_item.url # Not used yet

    logging.info(f"HTTP POST: Received data for client_id: {client_id}")

    if client_id not in manager.active_connections:
        logging.warning(f"Warning: Client {client_id} not connected via WebSocket. Cannot stream.")
        # raise HTTPException(status_code=404, detail=f"Client {client_id} not connected via WebSocket")
        return {"message": f"Client {client_id} not connected. Processing skipped."}

    queue = asyncio.Queue()

    # # Start the task that consumes from the queue and sends to the WebSocket
    # This task will run until it receives None from the queue or is canceled
    consumer_task = asyncio.create_task(stream_tokens_to_client(client_id, queue))
    manager.add_streaming_task(client_id, consumer_task) # register the task

    logging.info(f"HTTP POST: Dispatching LLM generation for {client_id}...")
    background_tasks.add_task(llm_service.run_async_stream, summary, 512, queue)

    logging.info(f"HTTP POST: Responding 202 Accepted to Service A for client {client_id}.")
    return {"message": "Data received, processing started, streaming initiated via WebSocket."}


@app.websocket("/api/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            logging.info(f"Received data from {client_id}: {data}")

            if data.get("action") == "generate_context":
                prompt = data.get("prompt")
                max_length = data.get("max_length", 512) # Default max_length

                if not prompt:
                    await manager.send_personal_message("ERROR: 'prompt' is required.", client_id)
                    continue

                # Create a queue for streaming tokens
                queue = asyncio.Queue()

                # LLM generating in background process (not blocking the WebSocket, concurrent)
                generation_task = asyncio.create_task(
                    llm_service.run_async_stream(prompt, max_length, queue)
                )
                logging.info(f"Task created for {client_id} prompt: {prompt[:30]}...")

                # Consume from the queue and send to the client via WebSocket
                while True:
                    token_or_signal = await queue.get()
                    if token_or_signal is None:
                        logging.info(f"End signal received for {client_id}.")
                        break
                    # Send token/chunk to the client
                    await manager.send_personal_message(token_or_signal, client_id)
                    queue.task_done()

                await generation_task
                logging.info(f"Generation task for {client_id} completed.")

            else:
                await manager.send_personal_message(f"Unknown action: {data.get('action')}", client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"Unexpected error for client {client_id}: {e}")
        manager.disconnect(client_id)
        try:
             await websocket.close(code=1011, reason=f"Internal server error: {e}")
        except:
             pass