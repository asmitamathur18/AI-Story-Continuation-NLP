"""Small local HTTP API to test the same inference code without cloud deployment."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from model import generate_story

app = FastAPI(title="AI Story Continuation API", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=60, ge=10, le=180)
    temperature: float = Field(default=0.8, ge=0.2, le=1.5)


@app.get("/health")
def health():
    return {"status": "ok", "model": "distilgpt2"}


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        text = generate_story(
            request.prompt, request.max_new_tokens, request.temperature
        )
        return {"generated_text": text}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc


if __name__ == "__main__":
    uvicorn.run("serve_local:app", host="127.0.0.1", port=8000, reload=False)
