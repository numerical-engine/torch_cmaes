import torch

def fix_cov(Cov:torch.Tensor, eps:float = 1e-8)->torch.Tensor:
    """Fix covariance matrix to be positive definite."""
    Cov = (Cov + Cov.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(Cov)
    eigvals = torch.clamp(eigvals, min=eps)

    Cov_fixed = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    return (Cov_fixed + Cov_fixed.T) / 2

def get_Covsqrt(Cov:torch.Tensor, eps:float = 1e-8)->torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(Cov)
    eigvals = torch.clamp(eigvals, min=eps)

    Covsqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    return (Covsqrt + Covsqrt.T) / 2