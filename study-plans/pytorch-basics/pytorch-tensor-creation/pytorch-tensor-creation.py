import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    match method:
        case "zeros":
            return torch.zeros(shape , dtype=torch.float32)
        case "ones":
            return torch.ones(shape , dtype=torch.float32)
        case "full":
            return torch.full(shape , value,dtype=torch.float32)
        case "eye":
            return torch.eye(*shape , dtype=torch.float32)
        case "arange":
            return torch.arange(shape[0] , shape[1] , 1 , dtype=torch.float32)
        case "linspace":
            return torch.linspace(shape[0] , shape[1] , shape[2] , dtype=torch.float32)
