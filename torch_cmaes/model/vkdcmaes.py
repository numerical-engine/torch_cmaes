import torch

from torch_cmaes import default_params, utils
from torch_cmaes.model.cmaes import CMA_ES


class VkD_CMA_ES(CMA_ES):
    def __init__(self, mean: torch.Tensor, Diag: torch.Tensor, k: int, sigma: float = 2.0, v: torch.Tensor | None = None, eps: float = 1e-8) -> None:
        assert mean.device == Diag.device
        assert Diag.dim() == 1, "Diag must be a 1D vector for VkD_CMA_ES"
        assert k >= 1, "k must be >= 1"

        self.sigma = sigma
        self.mean = mean
        self.p_sigma = torch.zeros_like(mean)
        self.p_c = torch.zeros_like(mean)

        self.k = min(k, mean.shape[0])
        self._eps = eps
        self.Diag = torch.clamp(Diag.clone(), min=self._eps)

        if v is None:
            self.v = torch.zeros(self.dim, self.k, device=mean.device, dtype=mean.dtype)
        else:
            assert v.shape == (self.dim, self.k), "v must have shape (dim, k)"
            self.v = v.clone().to(device=mean.device, dtype=mean.dtype)

    @property
    def Cov(self) -> torch.Tensor:
        eps = torch.finfo(self.mean.dtype).eps
        diag = torch.clamp(self.Diag, min=eps)
        D = torch.diag(diag)
        I = torch.eye(self.dim, device=self.device, dtype=self.mean.dtype)
        cov = D @ (I + self.v @ self.v.T) @ D
        return 0.5 * (cov + cov.T)

    @Cov.setter
    def Cov(self, Cov: torch.Tensor) -> None:
        eps = max(self._eps, torch.finfo(self.mean.dtype).eps)

        if Cov.dim() == 1:
            Cov = torch.diag(Cov)
        Cov = utils.fix_cov(Cov)

        diag_cov = torch.diag(Cov)
        off = Cov - torch.diag(diag_cov)
        eigvals, eigvecs = torch.linalg.eigh(off)
        if self.k == 1:
            idx = torch.argmax(eigvals)
            top = torch.clamp(eigvals[idx], min=0.0)
            vec = eigvecs[:, idx]
            if top.item() > 0:
                outer_v = (torch.sqrt(top + eps) * vec).view(-1, 1)
            else:
                outer_v = torch.zeros(self.dim, 1, device=self.device, dtype=self.mean.dtype)
        else:
            sorted_idx = torch.argsort(eigvals, descending=True)
            sel = sorted_idx[: self.k]
            top_vals = torch.clamp(eigvals[sel], min=0.0)
            top_vecs = eigvecs[:, sel]

            scales = torch.where(top_vals > 0, torch.sqrt(top_vals + eps), torch.zeros_like(top_vals))
            outer_v = top_vecs * scales.view(1, -1)

        residual_diag = torch.diag(Cov - outer_v @ outer_v.T)
        self.Diag = torch.sqrt(torch.clamp(residual_diag, min=eps))

        inv_diag = 1.0 / torch.clamp(self.Diag, min=eps)
        self.v = inv_diag.view(-1, 1) * outer_v

    def to(self, device: str) -> None:
        self.mean = self.mean.to(device)
        self.Diag = self.Diag.to(device)
        self.v = self.v.to(device)
        self.p_sigma = self.p_sigma.to(device)
        self.p_c = self.p_c.to(device)

    def sampling(self, fun: callable, **kwargs) -> list[torch.Tensor]:
        lam = kwargs.get("lam", default_params.lam(self.dim))

        diag_scale = torch.clamp(self.Diag, min=torch.finfo(self.mean.dtype).eps)
        z0 = torch.randn(lam, self.dim, device=self.device, dtype=self.mean.dtype)
        r = torch.randn(lam, self.k, device=self.device, dtype=self.mean.dtype)
        y = (z0 + r @ self.v.T) * diag_scale

        A = utils.get_Covsqrt(self.Cov)
        z = torch.linalg.solve(A, y.T).T

        samples = self.mean + self.sigma * y
        fitness = self.get_fitness(fun, samples, **kwargs)
        idx = torch.argsort(fitness)

        samples = samples[idx]
        y = y[idx]
        z = z[idx]

        return samples, y, z
