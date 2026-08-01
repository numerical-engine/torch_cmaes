import torch

from torch_cmaes import default_params, utils
from torch_cmaes.model.cmaes import CMA_ES

class VD_CMA_ES(CMA_ES):
	def __init__(self, mean:torch.Tensor, Diag:torch.Tensor, sigma:float = 2.0, v:torch.Tensor = None, eps:float = 1e-8)->None:
		assert mean.device == Diag.device
		self.sigma = sigma
		self.mean = mean
		self.p_sigma = torch.zeros_like(mean)
		self.p_c = torch.zeros_like(mean)
		self._eps = eps
		self.v = torch.zeros_like(mean) if v is None else v.clone().to(device=mean.device, dtype=mean.dtype)
		assert Diag.dim() == 1, "Diag must be a 1D vector for VD_CMA_ES"
		self.Diag = torch.clamp(Diag.clone(), min=self._eps)


	@property
	def Cov(self)->torch.Tensor:
		eps = torch.finfo(self.mean.dtype).eps
		diag = torch.clamp(self.Diag, min=eps)
		D = torch.diag(diag)
		I = torch.eye(self.dim, device=self.device, dtype=self.mean.dtype)
		cov = D @ (I + self.v.view(-1, 1) @ self.v.view(1, -1)) @ D
		return (cov + cov.T) / 2

	@Cov.setter
	def Cov(self, Cov:torch.Tensor)->None:
		eps = max(self._eps, torch.finfo(self.mean.dtype).eps)

		if Cov.dim() == 1:
			Cov = torch.diag(Cov)
		Cov = utils.fix_cov(Cov)
		diag_cov = torch.diag(Cov)
		off = Cov - torch.diag(diag_cov)

		eigvals, eigvecs = torch.linalg.eigh(off)
		idx = torch.argmax(eigvals)
		top = torch.clamp(eigvals[idx], min=0.0)
		vec = eigvecs[:, idx]

		outer_v = torch.zeros_like(self.mean)
		if top.item() > 0:
			outer_v = torch.sqrt(top + eps) * vec

		residual_diag = torch.diag(Cov - outer_v.view(-1, 1) @ outer_v.view(1, -1))
		self.Diag = torch.sqrt(torch.clamp(residual_diag, min=eps))
		self.v = outer_v / torch.clamp(self.Diag, min=eps)

	def to(self, device:str)->None:
		self.mean = self.mean.to(device)
		self.Diag = self.Diag.to(device)
		self.v = self.v.to(device)
		self.p_sigma = self.p_sigma.to(device)
		self.p_c = self.p_c.to(device)

	def sampling(self, fun:callable, **kwargs)->list[torch.Tensor]:
		lam = kwargs.get("lam", default_params.lam(self.dim))

		diag_std = torch.clamp(self.Diag, min=torch.finfo(self.mean.dtype).eps)
		z0 = torch.randn(lam, self.dim, device=self.device, dtype=self.mean.dtype)
		r = torch.randn(lam, 1, device=self.device, dtype=self.mean.dtype)
		y = (z0 + r * self.v) * diag_std

		A = utils.get_Covsqrt(self.Cov)
		z = torch.linalg.solve(A, y.T).T

		samples = self.mean + self.sigma * y
		fitness = self.get_fitness(fun, samples, **kwargs)
		idx = torch.argsort(fitness)

		samples = samples[idx]
		y = y[idx]
		z = z[idx]

		return samples, y, z