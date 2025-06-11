"""
KL Divergence Calculator for Normal Distributions

This module computes the Kullback-Leibler (KL) divergence between two normal distributions.
KL divergence measures how one probability distribution diverges from another.
"""

import math

def kl_normal(mu0, sigma0, mu1, sigma1):
    """
    Compute KL divergence KL( N(mu0, sigma0^2) || N(mu1, sigma1^2) ).
    
    Parameters:
    -----------
    mu0 : float
        Mean of the first normal distribution (p).
    sigma0 : float
        Standard deviation of the first normal distribution (p).
    mu1 : float
        Mean of the second normal distribution (q).
    sigma1 : float
        Standard deviation of the second normal distribution (q).
    
    Returns:
    --------
    float
        KL divergence value.
    
    Notes:
    ------
    The KL divergence is computed as:
    KL(p||q) = ln(sigma1/sigma0) + (sigma0^2 + (mu0 - mu1)^2) / (2 * sigma1^2) - 0.5
    """
    
    # Term 1: Log ratio of standard deviations
    term1 = math.log(sigma1 / sigma0)
    
    # Term 2: Sum of variances and squared mean difference, scaled by q's variance
    term2 = (sigma0**2 + (mu0 - mu1)**2) / (2 * sigma1**2)
    
    # Final KL divergence
    return term1 + term2 - 0.5


def kl_normal_symmetric(mu0, sigma0, mu1, sigma1):
    """
    Compute symmetric KL divergence between two normal distributions.
    
    Parameters:
    -----------
    mu0 : float
        Mean of the first normal distribution.
    sigma0 : float
        Standard deviation of the first normal distribution.
    mu1 : float
        Mean of the second normal distribution.
    sigma1 : float
        Standard deviation of the second normal distribution.
    
    Returns:
    --------
    float
        Symmetric KL divergence: 0.5 * (KL(p||q) + KL(q||p))
    """
    kl_pq = kl_normal(mu0, sigma0, mu1, sigma1)
    kl_qp = kl_normal(mu1, sigma1, mu0, sigma0)
    return 0.5 * (kl_pq + kl_qp)


# Example usage
if __name__ == "__main__":
    # Example 1: Asymmetric KL divergence
    mu0, sigma0 = 0.0, 1.0       # p ~ N(0,1)
    mu1, sigma1 = 1.0, 2.0       # q ~ N(1,4)
    kl = kl_normal(mu0, sigma0, mu1, sigma1)
    print(f"KL(p||q) = {kl:.6f}")
    
    # Example 2: Symmetric KL divergence
    kl_sym = kl_normal_symmetric(mu0, sigma0, mu1, sigma1)
    print(f"Symmetric KL(p,q) = {kl_sym:.6f}") 