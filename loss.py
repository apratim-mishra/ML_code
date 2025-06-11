"""
Loss Functions Implementation
Comprehensive collection of loss functions for machine learning tasks.
"""

import numpy as np
import math


class RegressionLosses:
    """
    Collection of regression loss functions.
    Used when predicting continuous values.
    """
    
    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error (MSE)
        
        When to use:
        - Standard regression loss
        - When you want to heavily penalize large errors
        - Smooth and easy to optimize
        - Sensitive to outliers
        
        Formula: L = (1/n) * Σ(y_i - ŷ_i)²
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean Absolute Error (MAE)
        
        When to use:
        - More robust to outliers than MSE
        - When you want equal penalty for all errors
        - Less sensitive to extreme values
        - Gradient discontinuous at zero
        
        Formula: L = (1/n) * Σ|y_i - ŷ_i|
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """
        Huber Loss
        
        When to use:
        - Best of both MSE and MAE
        - Robust to outliers while maintaining smoothness
        - Good for regression with some outliers
        - Combines quadratic loss for small errors, linear for large ones
        
        Formula: L_δ(r) = 0.5*r² if |r|≤δ, δ(|r|-0.5δ) if |r|>δ
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residual = np.abs(y_true - y_pred)
        
        # Quadratic for small residuals, linear for large ones
        loss = np.where(residual <= delta,
                       0.5 * residual ** 2,
                       delta * (residual - 0.5 * delta))
        return np.mean(loss)
    
    @staticmethod
    def log_cosh_loss(y_true, y_pred):
        """
        Log-Cosh Loss
        
        When to use:
        - Smooth like MSE but less punishing for big outliers
        - Approximates MAE for large errors, MSE for small errors
        - Good compromise between MSE and MAE
        
        Formula: L = Σ log(cosh(ŷ_i - y_i))
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        diff = y_pred - y_true
        return np.mean(np.log(np.cosh(diff)))


class ClassificationLosses:
    """
    Collection of classification loss functions.
    Used when predicting class labels or probabilities.
    """
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        """
        Binary Cross-Entropy (Logistic Loss)
        
        When to use:
        - Binary classification problems
        - When you have probabilistic outputs
        - Standard choice for binary classification
        
        Formula: L = -[y*log(p) + (1-y)*log(1-p)]
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        """
        Categorical Cross-Entropy
        
        When to use:
        - Multi-class classification
        - When classes are mutually exclusive
        - y_true should be one-hot encoded
        
        Formula: L = -Σ_k y_k * log(p_k)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def kl_divergence(y_true, y_pred, epsilon=1e-15):
        """
        Kullback-Leibler Divergence
        
        When to use:
        - Measuring difference between probability distributions
        - When both y_true and y_pred are probability distributions
        - Information theory applications
        
        Formula: KL(p||q) = Σ p(x) * ln(p(x)/q(x))
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip to prevent log(0)
        y_true = np.clip(y_true, epsilon, 1)
        y_pred = np.clip(y_pred, epsilon, 1)
        
        return np.sum(y_true * np.log(y_true / y_pred))
    
    @staticmethod
    def jensen_shannon_divergence(y_true, y_pred, epsilon=1e-15):
        """
        Jensen-Shannon Divergence
        
        When to use:
        - Symmetric version of KL divergence
        - More stable than KL divergence
        - Measuring similarity between distributions
        
        Formula: JS(p,q) = 0.5*KL(p||M) + 0.5*KL(q||M), M=(p+q)/2
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate mixture distribution
        M = 0.5 * (y_true + y_pred)
        
        # Calculate KL divergences
        kl1 = ClassificationLosses.kl_divergence(y_true, M, epsilon)
        kl2 = ClassificationLosses.kl_divergence(y_pred, M, epsilon)
        
        return 0.5 * kl1 + 0.5 * kl2
    
    @staticmethod
    def focal_loss(y_true, y_pred, alpha=1.0, gamma=2.0, epsilon=1e-15):
        """
        Focal Loss
        
        When to use:
        - Imbalanced datasets
        - When you want to focus on hard examples
        - Object detection and dense prediction tasks
        
        Formula: L = -α(1-p)^γ * log(p)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate focal loss
        pt = np.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = alpha * (1 - pt) ** gamma
        loss = -focal_weight * np.log(pt)
        
        return np.mean(loss)
    
    @staticmethod
    def hinge_loss(y_true, y_pred):
        """
        Hinge Loss
        
        When to use:
        - Support Vector Machines (SVMs)
        - Maximum margin classification
        - y_true should be in {-1, +1}
        
        Formula: L = max(0, 1 - y * ŝ)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    @staticmethod
    def squared_hinge_loss(y_true, y_pred):
        """
        Squared Hinge Loss
        
        When to use:
        - Smoother version of hinge loss
        - When you want differentiability everywhere
        - Alternative to standard hinge for SVMs
        
        Formula: L = (max(0, 1 - y * ŝ))²
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        hinge = np.maximum(0, 1 - y_true * y_pred)
        return np.mean(hinge ** 2)
    
    @staticmethod
    def logistic_margin_loss(y_true, y_pred):
        """
        Logistic-Margin Loss
        
        When to use:
        - Smooth approximation to hinge loss
        - Logistic regression with margin
        - When you need smooth gradients
        
        Formula: L = log(1 + exp(-y * ŝ))
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return np.mean(np.log(1 + np.exp(-y_true * y_pred)))


class RankingLosses:
    """
    Collection of ranking and pairwise loss functions.
    Used for learning to rank and similarity learning tasks.
    """
    
    @staticmethod
    def pairwise_hinge_loss(scores_i, scores_j, margin=1.0):
        """
        Pairwise Hinge Loss (RankNet style)
        
        When to use:
        - Learning to rank problems
        - When you want item i ranked above item j
        - Information retrieval systems
        
        Formula: L = max(0, margin - (ŝ_i - ŝ_j))
        """
        scores_i = np.array(scores_i)
        scores_j = np.array(scores_j)
        
        return np.mean(np.maximum(0, margin - (scores_i - scores_j)))
    
    @staticmethod
    def triplet_loss(anchor, positive, negative, margin=1.0):
        """
        Triplet Loss
        
        When to use:
        - Face recognition and verification
        - Learning embeddings where similarity matters
        - Metric learning problems
        
        Formula: L = max(0, d(a,p) - d(a,n) + α)
        """
        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)
        
        # Calculate distances (L2 norm)
        dist_pos = np.sum((anchor - positive) ** 2, axis=1)
        dist_neg = np.sum((anchor - negative) ** 2, axis=1)
        
        return np.mean(np.maximum(0, dist_pos - dist_neg + margin))
    
    @staticmethod
    def contrastive_loss(x1, x2, y, margin=1.0):
        """
        Contrastive Loss
        
        When to use:
        - Siamese networks
        - Learning embeddings for similarity
        - When you have pairs of similar/dissimilar examples
        
        Formula: L = y*d² + (1-y)*max(0,m-d)²
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)
        
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        
        # Contrastive loss
        loss = y * distance ** 2 + (1 - y) * np.maximum(0, margin - distance) ** 2
        
        return np.mean(loss)


class LossCalculator:
    """
    Main loss calculator class that provides easy access to all loss functions.
    """
    
    def __init__(self):
        self.regression = RegressionLosses()
        self.classification = ClassificationLosses()
        self.ranking = RankingLosses()
    
    def compute_loss(self, loss_type, y_true, y_pred, **kwargs):
        """
        Compute loss based on specified type.
        
        Parameters:
        -----------
        loss_type : str
            Type of loss to compute
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        **kwargs : dict
            Additional parameters for specific loss functions
        """
        
        # Regression losses
        if loss_type == 'mse':
            return self.regression.mse(y_true, y_pred)
        elif loss_type == 'mae':
            return self.regression.mae(y_true, y_pred)
        elif loss_type == 'huber':
            return self.regression.huber_loss(y_true, y_pred, **kwargs)
        elif loss_type == 'log_cosh':
            return self.regression.log_cosh_loss(y_true, y_pred)
        
        # Classification losses
        elif loss_type == 'binary_cross_entropy':
            return self.classification.binary_cross_entropy(y_true, y_pred, **kwargs)
        elif loss_type == 'categorical_cross_entropy':
            return self.classification.categorical_cross_entropy(y_true, y_pred, **kwargs)
        elif loss_type == 'kl_divergence':
            return self.classification.kl_divergence(y_true, y_pred, **kwargs)
        elif loss_type == 'js_divergence':
            return self.classification.jensen_shannon_divergence(y_true, y_pred, **kwargs)
        elif loss_type == 'focal':
            return self.classification.focal_loss(y_true, y_pred, **kwargs)
        elif loss_type == 'hinge':
            return self.classification.hinge_loss(y_true, y_pred)
        elif loss_type == 'squared_hinge':
            return self.classification.squared_hinge_loss(y_true, y_pred)
        elif loss_type == 'logistic_margin':
            return self.classification.logistic_margin_loss(y_true, y_pred)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    print("Loss Functions Implementation Test")
    print("=" * 50)
    
    # Create loss calculator
    loss_calc = LossCalculator()
    
    # Test regression losses
    print("\n1. REGRESSION LOSSES:")
    print("-" * 30)
    y_true_reg = [3.0, -0.5, 2.0, 7.0]
    y_pred_reg = [2.5, 0.0, 2.0, 8.0]
    
    print(f"True values: {y_true_reg}")
    print(f"Predictions: {y_pred_reg}")
    print(f"MSE Loss: {loss_calc.compute_loss('mse', y_true_reg, y_pred_reg):.4f}")
    print(f"MAE Loss: {loss_calc.compute_loss('mae', y_true_reg, y_pred_reg):.4f}")
    print(f"Huber Loss: {loss_calc.compute_loss('huber', y_true_reg, y_pred_reg):.4f}")
    print(f"Log-Cosh Loss: {loss_calc.compute_loss('log_cosh', y_true_reg, y_pred_reg):.4f}")
    
    # Test classification losses
    print("\n2. CLASSIFICATION LOSSES:")
    print("-" * 30)
    y_true_cls = [1, 0, 1, 1]
    y_pred_cls = [0.9, 0.1, 0.8, 0.7]
    
    print(f"True labels: {y_true_cls}")
    print(f"Predicted probabilities: {y_pred_cls}")
    print(f"Binary Cross-Entropy: {loss_calc.compute_loss('binary_cross_entropy', y_true_cls, y_pred_cls):.4f}")
    print(f"Focal Loss: {loss_calc.compute_loss('focal', y_true_cls, y_pred_cls):.4f}")
    
    # Test hinge loss (requires labels in {-1, +1})
    y_true_hinge = [1, -1, 1, 1]
    y_pred_hinge = [0.8, -0.2, 0.9, 0.6]
    print(f"Hinge Loss: {loss_calc.compute_loss('hinge', y_true_hinge, y_pred_hinge):.4f}")
    
    # Test ranking losses
    print("\n3. RANKING LOSSES:")
    print("-" * 30)
    anchor = [[1, 2], [3, 4]]
    positive = [[1.1, 2.1], [3.1, 4.1]]  # Similar to anchor
    negative = [[5, 6], [7, 8]]  # Dissimilar to anchor
    
    triplet_loss_val = loss_calc.ranking.triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {triplet_loss_val:.4f}")
    
    # Test contrastive loss
    x1 = [[1, 2], [3, 4]]
    x2 = [[1.1, 2.1], [7, 8]]
    y_pairs = [1, 0]  # 1 = similar, 0 = dissimilar
    
    contrastive_loss_val = loss_calc.ranking.contrastive_loss(x1, x2, y_pairs)
    print(f"Contrastive Loss: {contrastive_loss_val:.4f}")
    
    print(f"\n{'='*50}")
    print("All loss functions tested successfully!")
    print(f"{'='*50}") 