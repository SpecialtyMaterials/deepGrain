import torch
import torch.nn as nn

def load_checkpoint(filepath):
    return torch.load(filepath, map_location='cpu')

def save_checkpoint(data, filepath):
    torch.save(data, filepath)

def reshape_tensor(tensor, target_shape):
    # Ensure the number of elements matches
    if tensor.numel() != torch.prod(torch.tensor(target_shape)):
        raise ValueError("The total number of elements must match for reshaping.")
    return tensor.view(target_shape)

def adjust_checkpoint_shape(checkpoint, param_name, target_shape):
    if param_name in checkpoint:
        checkpoint[param_name] = reshape_tensor(checkpoint[param_name], target_shape)
    else:
        raise KeyError(f'Parameter {param_name} not found in checkpoint')
    return checkpoint

# Load the checkpoint
checkpoint_path =  "C:/Users/nickb/Desktop/projects/cobDetection/DiffusionEdge-main/weights/model-6.pt"
checkpoint = load_checkpoint(checkpoint_path)

# Adjust the shape of the specific parameter
param_name = 'model.decouple1.2.complex_weight'
target_shape = torch.Size([512, 10, 6, 2])
checkpoint = adjust_checkpoint_shape(checkpoint, param_name, target_shape)

# Save the modified checkpoint
modified_checkpoint_path = "C:/Users/nickb/Desktop/projects/cobDetection/DiffusionEdge-main/weights/modifiedWeights.pt"
save_checkpoint(checkpoint, modified_checkpoint_path)

print(f"Checkpoint successfully adjusted and saved to {modified_checkpoint_path}")