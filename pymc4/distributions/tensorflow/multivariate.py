"""PyMC4 multivariate random variables for tensorflow."""
import tensorflow_probability as tfp
from pymc4.distributions import abstract
from pymc4.distributions.tensorflow.distribution import BackendDistribution


tfd = tfp.distributions

__all__ = [
    "Dirichlet",
    "LKJCorr",
    "LKJCholeskyCov",
    "MvNormal",
]


class Dirichlet(BackendDistribution, abstract.Dirichlet):
    def _init_backend(self):
        a = self.conditions["a"]
        self._backend_distribution = tfd.Dirichlet(concentration=a)


class LKJCorr(BackendDistribution, abstract.LKJCorr):
    def _init_backend(self):
        n, eta = self.conditions["n"], self.conditions["eta"]
        self._backend_distribution = tfd.LKJ(dimension=n, concentration=eta)


class LKJCholeskyCov(BackendDistribution, abstract.LKJCholeskyCov):
    def _init_backend(self):
        n, eta = self.conditions["n"], self.conditions["eta"]
        self._backend_distribution = tfd.CholeskyLKJ(dimension=n, concentration=eta)


class MvNormal(BackendDistribution, abstract.MvNormal):
    def _init_backend(self):
        mu = self.conditions["mu"]
        assert "chol" in self.conditions != "cov" in self.conditions, "exactly one of chol or cov must be specified."
        if "chol" in self.conditions:, 
            chol = self.conditions["chol"]
            self._backend_distribution = tfd.MultivariateNormalTriL(loc=mu, scale_tril=chol)
        elif "cov" in self.conditions:
            cov = self.conditions["cov"]
            self._backend_distribution = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)

