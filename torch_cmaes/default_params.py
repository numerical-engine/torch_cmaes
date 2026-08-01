import math
import torch

def lam(dim:int)->int:
    return 4 + int(3 * math.log(dim))

def mu(lam:int)->int:
    return lam // 2

def omega(lam:int, mu:int, device:str, dtype:str = torch.float)->torch.Tensor:
    w = torch.zeros(lam, device=device, dtype=dtype)
    w[:mu] = torch.tensor([math.log((lam + 1) / 2) - math.log(i + 1) for i in range(mu)], device=device, dtype=dtype)

    w /= torch.sum(w)
    return w.view(-1, 1)

def mu_eff(omega:torch.Tensor, mu:int)->float:
    return 1 / torch.sum(omega[:mu]**2)

def c_sigma(mu_eff:float, dim:int)->float:
    return (mu_eff + 2) / (dim + mu_eff + 5)

def c_c(mu_eff:float, dim:int)->float:
    return (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)

def c_m()->int:
    return 1

def c_1(mu_eff:float, dim:int, lam:int)->float:
    alpha = min(2, lam/3)
    return alpha / ((dim + 1.3)**2 + mu_eff)

def c_mu(mu_eff:float, dim:int, lam:int)->float:
    alpha = min(2, lam/3)
    c = c_1(mu_eff, dim, lam)
    return min(1 - c, alpha * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + alpha * mu_eff / 2))

def d_sigma(mu_eff:float, dim:int)->float:
    c = c_sigma(mu_eff, dim)
    return 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c