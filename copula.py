import numpy as np
import pandas as pd
from scipy import stats

class EmpiricalCopula:
    def __init__(self, data):
        """
        Fits a Gaussian Copula using empirical marginals.

        Args:
            data (numpy.ndarray): Each row is an observation with columns being variables.
        """
        self.data = data
        
        # Compute empirical CDFs
        self.cdfs = [stats.ecdf(marginal[~np.isnan(marginal)]).cdf for marginal in data.T]
        
        # Map data to 0-1 space
        values = [
            np.where(np.isnan(marginal), marginal, cdf.evaluate(marginal))
            for marginal, cdf in zip(data.T, self.cdfs)
        ]
        
        # Get correlation structure
        self.corr = pd.DataFrame(values).T.corr().fillna(0).values
        
        # Get sorted data for inverse CDFs
        self.sorted_data = [sorted(col[~np.isnan(col)]) for col in data.T]
        
        # Check if discrete
        self.is_discrete = [len(np.unique(marginal)) < 10 for marginal in self.data.T]

    def sample(self, num_samples=10):
        """
        Generate synthetic samples using the fitted copula.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            numpy.ndarray: Synthetic samples in the original data space.
        """
        # Generate random samples with correlation structure
        normal_vec = np.random.multivariate_normal(
            np.zeros(self.corr.shape[0]), 
            self.corr, 
            size=num_samples
        )
        
        # Map to quantile space
        quantile_vec = stats.norm.cdf(normal_vec)
        
        # Map back to original space
        synth = np.array([
            np.quantile(
                self.sorted_data[i], 
                samples, 
                method='closest_observation' if self.is_discrete[i] else 'linear'
            )
            for i, samples in enumerate(quantile_vec.T)
        ]).T
        
        return synth