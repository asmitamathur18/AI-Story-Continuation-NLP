# AI Story Continuation System

A beginner-friendly NLP mini-project using DistilGPT2, PyTorch, Hugging Face Transformers,
TorchServe, Streamlit, and an optional Amazon SageMaker deployment.

## What is included
- `app.py`: Streamlit UI (prompt, length, temperature)
- `model.py`: shared DistilGPT2 loading and generation logic
- `handler.py`: TorchServe custom inference handler
- `serve_local.py`: simple local HTTP API for quick testing
- `config.properties`: TorchServe configuration
- `model-store/`: place the generated `.mar` model archive here
- `scripts/`: model packaging and SageMaker deployment helpers
- `tests/`: basic input-validation tests

## 1. Requirements
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended for local CPU testing
- For AWS deployment: an AWS account, permissions for SageMaker, ECR, and S3,
  and faculty-approved AWS credits/account

## 2. Create environment and install packages

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Run the UI locally (direct PyTorch inference)
```powershell
streamlit run app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).
The first run downloads DistilGPT2 from Hugging Face, so internet access is needed.

## 4. Run the standalone local API
In a second terminal, activate the same environment:
```powershell
python serve_local.py
```
API health check:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```
Generate:
```powershell
$body = @{
  prompt = "One evening, a student found a mysterious key"
  max_new_tokens = 60
  temperature = 0.8
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body $body
```

## 5. Run through TorchServe locally

TorchServe has compatibility constraints and is in limited maintenance. Use a compatible
Python/PyTorch/TorchServe combination and check the official TorchServe compatibility notes
before installation. The supplied handler is the same generation implementation used by
the Streamlit app.

Install TorchServe in a separate compatible environment if required:
```powershell
pip install torchserve torch-model-archiver
```

Package the model:
```powershell
mkdir model-store
torch-model-archiver --model-name story_generator --version 1.0 `
  --handler handler.py --extra-files "model.py,requirements.txt" `
  --export-path model-store --force
```

Start TorchServe:
```powershell
torchserve --start --ncs --model-store model-store --models story_generator=story_generator.mar --ts-config config.properties
```

Test:
```powershell
$body = '{"prompt":"A robot arrived at a small village","max_new_tokens":50,"temperature":0.8}'
Invoke-RestMethod -Uri http://127.0.0.1:8080/predictions/story_generator -Method Post -ContentType "application/json" -Body $body
```

Stop:
```powershell
torchserve --stop
```

Note: model-archiver packaging and TorchServe dependency installation can vary by OS/version.
If the handler cannot import a package in the TorchServe worker, install it in the same
environment used to launch TorchServe and rebuild the `.mar` archive.

## 6. Amazon SageMaker deployment

The `sagemaker/` folder contains a Dockerfile and inference entrypoint for a custom
SageMaker real-time endpoint. It uses the same `model.py` generation code. This is a
separate deployment route from the local TorchServe demo: SageMaker custom containers
must implement the SageMaker inference-container contract. The provided container exposes
`/ping` and `/invocations` and runs the model with PyTorch/Transformers.

For a TorchServe-on-SageMaker demonstration, use a SageMaker-compatible TorchServe
container/configuration and explicitly verify the model server used by the endpoint.
Do not assume that a generic SageMaker PyTorch image automatically runs TorchServe.

See `sagemaker/README.md` for the deployment outline. Cloud resources can incur charges.
Delete the endpoint and endpoint configuration after the demonstration.

## 7. Project test checklist
- [ ] Generate continuations for at least 10 story prompts.
- [ ] Test empty prompt.
- [ ] Test an excessively long prompt.
- [ ] Compare short and long `max_new_tokens` settings.
- [ ] Compare low and high temperature.
- [ ] Record local PyTorch, TorchServe, and SageMaker outputs.
- [ ] Record latency and endpoint configuration.
- [ ] Delete AWS endpoint/resources after demonstration.

## Important limitation
DistilGPT2 is a general pretrained text-generation model, not a fact-checking or instruction-following
system. It can produce repetitive, incomplete, or nonsensical continuations. Outputs are creative
suggestions and should be reviewed.
