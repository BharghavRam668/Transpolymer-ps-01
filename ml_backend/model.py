import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import os
import json

class PolymerRegressor(nn.Module):
    def __init__(self, encoder, dropout_rate=0.3):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.regression_head(pooled_output)

def load_models(model_dir="regression_models", device="cpu"):
    models = {}
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load tokenizer and encoder once (shared across models)
    pretrained_model = AutoModelForMaskedLM.from_pretrained("merged_chemberta")
    encoder = pretrained_model.roberta.to(device)
    
    for prop_name in config.keys():
        # Initialize regressor
        regressor = PolymerRegressor(encoder).to(device)
        
        # Load weights
        weights_path = os.path.join(model_dir, f"{prop_name}_weights.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            regressor.load_state_dict(state_dict)
            regressor.eval()
            # Attach config for denormalization
            regressor.config = config[prop_name]
            models[prop_name] = regressor
        else:
            print(f"Warning: Weights not found for {prop_name} at {weights_path}")
    
    return models

if __name__ == "__main__":
    models = load_models()
    print("Models loaded:", list(models.keys()))