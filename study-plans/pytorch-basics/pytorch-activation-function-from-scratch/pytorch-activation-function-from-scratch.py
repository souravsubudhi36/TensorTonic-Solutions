import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    match method:
        case "relu":
            return torch.clamp(x , min=0.0)
        case "sigmoid":
            denominator = 1.0 + torch.exp(-x)
            return 1.0/denominator
        case "tanh":
            return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0
        case "leaky_relu":
            alpha = 0.01
            output = torch.where(x > 0, x , alpha*x)
            return output
        
            
        
