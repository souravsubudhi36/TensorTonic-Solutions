import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    match op:
        case "flatten":
            return torch.flatten(x)
        case "squeeze":
            return torch.squeeze(x)
        case 'transpose':
            return torch.transpose(x , 0, 1)
            
