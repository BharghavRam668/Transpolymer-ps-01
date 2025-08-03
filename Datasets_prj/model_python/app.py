from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
FINGERPRINT_SIZE = 2048  # Must match what was used during training
RADIUS = 2

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, 
                 hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

def load_models() -> Dict[str, Any]:
    """Load all models from the regression_models directory"""
    models = {}
    BASE_DIR = Path(__file__).parent.resolve()
    MODELS_DIR = BASE_DIR / "regression_models"
    
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found at {MODELS_DIR}")

    try:
        with open(MODELS_DIR / "config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

    # Default model architecture parameters
    DEFAULT_CONFIG = {
        "input_dim": FINGERPRINT_SIZE,
        "embed_dim": 256,
        "num_heads": 4,
        "hidden_dim": 512,
        "num_layers": 3,
        "dropout": 0.1
    }

    for prop, prop_config in config.items():
        model_path = MODELS_DIR / f"{prop}_weights.pt"
        try:
            if not model_path.exists():
                logger.warning(f"Model file not found for {prop}: {model_path}")
                continue

            # Merge configs with property-specific overriding defaults
            model_config = {**DEFAULT_CONFIG, **prop_config}
            
            model = TransformerModel(
                input_dim=model_config["input_dim"],
                embed_dim=model_config["embed_dim"],
                num_heads=model_config["num_heads"],
                hidden_dim=model_config["hidden_dim"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"]
            )

            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()

            models[prop] = {
                "model": model,
                "mean": prop_config.get("target_mean", 0),
                "std": prop_config.get("target_std", 1),
                "name": prop_config.get("name", prop),
                "unit": prop_config.get("unit", "")
            }
            logger.info(f"Successfully loaded model for {prop}")

        except Exception as e:
            logger.error(f"Error loading model {prop}: {e}")
            continue

    if not models:
        raise ValueError("No models were successfully loaded")
    
    return models

# Load models at startup
try:
    models = load_models()
    logger.info(f"Loaded models for: {list(models.keys())}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

class SMILESRequest(BaseModel):
    smiles: str

def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint using the new generator API"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Using the new MorganGenerator API
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 
        radius=RADIUS, 
        nBits=FINGERPRINT_SIZE,
        useChirality=True
    )
    return np.array(fp, dtype=np.float32)

@app.post("/predict")
async def predict(request: SMILESRequest):
    try:
        logger.info(f"Predicting properties for SMILES: {request.smiles}")
        
        # Convert SMILES to fingerprint
        fingerprint = smiles_to_fingerprint(request.smiles)
        input_tensor = torch.tensor([fingerprint], dtype=torch.float32)

        results = []
        for prop, data in models.items():
            try:
                with torch.no_grad():
                    # Get prediction and denormalize
                    pred = data["model"](input_tensor).item()
                    denorm_pred = pred * data["std"] + data["mean"]
                    
                    results.append({
                        "id": prop,
                        "name": data["name"],
                        "value": round(denorm_pred, 4),
                        "unit": data["unit"]
                    })
                    logger.info(f"Predicted {prop}: {denorm_pred}")
                    
            except Exception as e:
                logger.error(f"Error predicting {prop}: {e}")
                continue

        if not results:
            raise HTTPException(
                status_code=400,
                detail="No predictions generated - check model inputs and configuration"
            )

        return {"predictions": results}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

@app.get("/properties")
async def get_properties():
    """List all available properties with metadata"""
    return {
        "properties": [
            {
                "id": prop,
                "name": data["name"],
                "unit": data["unit"],
                "mean": data["mean"],
                "std": data["std"]
            }
            for prop, data in models.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")