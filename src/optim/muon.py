import torch
from torch import Tensor

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    """
    assert G.ndim >= 2  # Ensure we are dealing with a 2D or higher tensor 
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)  # Convert to bfloat16 if needed
    if G.size(-2) > G.size(-1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz

    Internally runs standard SGD with an orthogonalization post-processing step.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        for p in self.param_groups[0]["params"]:  # Since no grouping, all params are in param_groups[0]
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            original_shape = g.shape
            # Initialize momentum buffer if it doesn't exist
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            
            buf = state["momentum_buffer"]
            
            # Apply momentum
            buf.lerp_(g, 1 - self.param_groups[0]["momentum"])
            g = g.lerp_(buf, self.param_groups[0]["momentum"]) if self.param_groups[0]["nesterov"] else buf
            # For the case of convolutional filters or other 4D tensors, flatten them
            if g.ndim == 4:  # Handle 4D tensors (convolutional filters)
                g = g.view(len(g), -1)
            # Apply Newton-Schulz iteration for orthogonalization
            g = zeropower_via_newtonschulz5(g, steps=self.param_groups[0]["ns_steps"])
            # Apply weight decay (L2 regularization)
            p.mul_(1 - self.param_groups[0]["lr"] * self.param_groups[0]["weight_decay"])
            # Apply gradient update with the learning rate
            g = g.view(original_shape)
            p.add_(g, alpha=-self.param_groups[0]["lr"])


