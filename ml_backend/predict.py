import torch
from transformers import AutoTokenizer
from model import load_models
from rdkit import Chem

# Load tokenizer and models only once
tokenizer = AutoTokenizer.from_pretrained("merged_chemberta")
models = load_models()

def is_valid_smiles(smiles):
    """Check if the SMILES string is chemically valid using RDKit."""
    return Chem.MolFromSmiles(smiles) is not None

def predict_polymer_properties(smiles):
    if not smiles or not isinstance(smiles, str):
        raise ValueError("Input must be a non-empty SMILES string")
    
    if not is_valid_smiles(smiles):
        raise ValueError("Invalid SMILES string. Please enter a valid structure.")
    
    # Tokenize input
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Get device from one of the loaded models
    device = next(iter(models.values())).encoder.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    results = {}
    for model_name, model in models.items():
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
            normalized_value = output.item()
            
            # Denormalize
            target_mean = model.config["target_mean"]
            target_std = model.config["target_std"]
            predicted_value = normalized_value * target_std + target_mean
            
            results[model_name] = round(predicted_value, 3)

    return results
