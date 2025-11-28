# LLM tiny project

## Environment

Developed in a venv environment.

## Libraries

Needed libs :

- transformers
- datasets
- accelerate
- sentencepiece
- torch

## Commands to init project

### Linux

```bash
python -m venv venv
source venv/bin/activate
pip install transformers datasets accelerate sentencepiece
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install transformers datasets accelerate sentencepiece
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
