"""Main entrypoint for Auto Viral Cuts FastAPI & Gradio application."""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
from dotenv import load_dotenv

from src.api.routes import router as api_router
from src.ui.gradio_app import create_demo

load_dotenv()

app = FastAPI(
    title="Auto Viral Cuts API",
    description="API profissional para geração automática de cortes virais com Google Gemini e FFmpeg.",
    version="0.1.0",
)

# Include REST API routers
app.include_router(api_router)

# Mount client-side WebCodecs renderer studio
client_dir = os.path.join(os.path.dirname(__file__), "ui", "client_renderer")
if os.path.exists(client_dir):
    app.mount("/client", StaticFiles(directory=client_dir, html=True), name="client_renderer")


@app.get("/")
def root_redirect():
    """Root endpoint info."""
    return {
        "message": "Bem-vindo à API do Auto Viral Cuts!",
        "client": "/client",
        "ui": "/ui",
        "docs": "/docs",
    }


# Create Gradio demo and mount on FastAPI app
gradio_app = create_demo()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"

    print(f"🚀 Iniciando Auto Viral Cuts em http://{host}:{port}")
    print(f"⚡ Studio WebCodecs (Cliente) disponível em http://{host}:{port}/client")
    print(f"📱 Interface Servidor (Gradio) disponível em http://{host}:{port}/ui")
    print(f"📚 Documentação OpenAPI disponível em http://{host}:{port}/docs")


    uvicorn.run("src.main:app", host=host, port=port, reload=debug)
