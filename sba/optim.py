"""First-order SAM in trainable adapter coordinates, with one base Adam step.

Frozen CLIP is excluded. Exact restoration prevents perturbation accumulation.
Standard SAM (Foret et al., ICLR 2021), not a new optimizer claim.
"""
import math
import torch


def sam_step(parameters, optimizer, closure, rho=.05, max_grad_norm=5.):
    parameters=list(parameters)
    if not parameters or not math.isfinite(rho) or rho<0:
        raise ValueError('Nonempty parameters and finite nonnegative radius required')
    def backward():
        optimizer.zero_grad(set_to_none=True)
        loss=closure()
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError('Finite scalar objective required')
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError('Finite gradients for every adapter required')
        return float(loss.detach())
    base_loss=backward()
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
    originals=[p.detach().clone() for p in parameters]
    perturb_norm=0.
    try:
        with torch.no_grad():
            actual=[]
            for p,original in zip(parameters,originals):
                p.add_(p.grad*(rho/(norm+1e-12)))
                actual.append(torch.linalg.vector_norm(p-original))
            perturb_norm=float(torch.linalg.vector_norm(torch.stack(actual)))
        perturbed_loss=backward()
    finally:
        with torch.no_grad():
            for p,original in zip(parameters,originals): p.copy_(original)
    final_norm=float(torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True))
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in parameters):
        raise ValueError('Nonfinite updated parameters')
    return dict(ce=base_loss,perturbed_ce=perturbed_loss,loss=perturbed_loss,
                perturbation_norm=perturb_norm,gradient_norm=float(norm),second_gradient_norm=final_norm)
