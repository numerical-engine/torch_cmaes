import torch
import math

from torch_cmaes import default_params, utils

class CMA_ES:
    def __init__(self, mean:torch.Tensor, Cov:torch.Tensor, sigma:float = 2.0)->None:
        assert mean.device == Cov.device
        self.sigma = sigma
        self.mean = mean
        self.Cov = Cov
        self.p_sigma = torch.zeros_like(mean)
        self.p_c = torch.zeros_like(mean)

    @property
    def device(self):
        return self.mean.device

    @property
    def dim(self)->int:
        return self.mean.shape[0]

    def to(self, device:str)->None:
        self.mean = self.mean.to(device)
        self.Cov = self.Cov.to(device)
        self.p_sigma = self.p_sigma.to(device)
        self.p_c = self.p_c.to(device)

    def get_fitness(self, fun:callable, samples:torch.Tensor, **kwargs)->torch.Tensor:
        fitness = torch.tensor([fun(sample) for sample in samples], device=samples.device)
        return fitness

    def sampling(self, fun:callable, **kwargs)->list[torch.Tensor]:
        lam = kwargs.get("lam", default_params.lam(self.dim))

        z = torch.randn(lam, self.dim, device=self.device, dtype=self.mean.dtype)
        A = utils.get_Covsqrt(self.Cov)
        y = z @ A.T

        samples = self.mean + self.sigma * y
        fitness = self.get_fitness(fun, samples, **kwargs)
        idx = torch.argsort(fitness)

        samples = samples[idx]
        y = y[idx]
        z = z[idx]

        return samples, y, z

    def update_params(self, samples:torch.Tensor, y:torch.Tensor, z:torch.Tensor, **kwargs)->None:
        mu = kwargs.get("mu", default_params.mu(samples.shape[0]))
        omega = kwargs.get("omega", default_params.omega(samples.shape[0], mu, self.device, self.mean.dtype))
        mu_eff = kwargs.get("mu_eff", default_params.mu_eff(omega, mu))
        c_sigma = kwargs.get("c_sigma", default_params.c_sigma(mu_eff, self.dim))
        c_c = kwargs.get("c_c", default_params.c_c(mu_eff, self.dim))
        c_m = kwargs.get("c_m", default_params.c_m())
        d_sigma = kwargs.get("d_sigma", default_params.d_sigma(mu_eff, self.dim))
        c_1 = kwargs.get("c_1", default_params.c_1(mu_eff, self.dim, samples.shape[0]))
        c_mu = kwargs.get("c_mu", default_params.c_mu(mu_eff, self.dim, samples.shape[0]))

        dz = torch.sum(omega*z, dim=0)
        dy = torch.sum(omega*y, dim=0)

        self.p_sigma = (1. - c_sigma) * self.p_sigma + torch.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * dz
        self.p_c = (1. - c_c) * self.p_c + torch.sqrt(c_c * (2 - c_c) * mu_eff) * dy

        self.mean = self.mean + c_m * self.sigma * dy
        norm = math.sqrt(self.dim)*(1. - 1./(4*self.dim) + 1./(21*self.dim**2))
        self.sigma = self.sigma*torch.exp((c_sigma / d_sigma) * (torch.norm(self.p_sigma) / norm - 1))

        cov_next = self.Cov + c_1 * (self.p_c.view(-1, 1) @ self.p_c.view(1, -1) - self.Cov) + c_mu * (y[:mu].T @ (omega[:mu] * y[:mu]) - self.Cov)
        self.Cov = utils.fix_cov(cov_next)



    def step(self, fun:callable, **kwargs)->None:
        samples, y, z = self.sampling(fun, **kwargs)
        self.update_params(samples, y, z, **kwargs)