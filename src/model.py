"""

model.py

Logistic regression implementation from scratch.

Provides a simple logistic regression model trained with batch gradient descent.
Supports optional L2 regularization (Ridge) to reduce overfitting.
"""
import numpy as np

class LogisticRegressionScratch:
    """ Linear regression model trained using gradient descent. """
    def __init__(self, alpha = 0.01, num_iters = 1000, l2_lambda=0.0, class_weight=None):
        """
        Parameters
        ----------
        alpha : float
            Learning rate.
        num_iters : int
            Number of gradient descent iterations.
        l2_lambda : float
            L2 regularization strength (0 disables regularization).
        """
        self.w = None
        self.b = None
        self.alpha = alpha
        self.num_iters = num_iters
        self.cost_history = []
        self.l2_lambda = l2_lambda
        self.class_weight = class_weight

    def fit(self, X, y):
        """ Train model parameters on the given training data. """
        m, n = X.shape

        self.w = np.zeros(n) # Initialize weights to zero
        self.b = 0.0 # Initialize bias
        self.cost_history = []

        eps = 1e-15

        if self.class_weight == "balanced": # Balanced class weighting
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)

            w_pos = m / (2 * n_pos)
            w_neg = m / (2 * n_neg)

            sample_weights = np.where(y == 1, w_pos, w_neg) # higher weight to minority class
        else:
            sample_weights = np.ones(m) # no class weighting

        for _ in range(self.num_iters):
            z = X @ self.w + self.b
            sigmoid = 1 / (1 + np.exp(-z))
            sigmoid = np.clip(sigmoid, eps, 1 - eps)  # Avoid log(0) in loss
            error = (sigmoid - y) * sample_weights

            d_w = (1 / m) * (X.T @ (error))
            d_b = (1 / m) * np.sum(error)

            if self.l2_lambda > 0: # Add L2 regularization to weight gradient (does not apply to bias)
                d_w += (self.l2_lambda / m) * self.w

            self.w -= self.alpha * d_w
            self.b -= self.alpha * d_b
            
            cost = -(1 / m) * np.sum(sample_weights * (y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid)))

            if self.l2_lambda > 0: # Add L2 penalty to cost (bias is not regularized)
                cost += (self.l2_lambda / (2 * m)) * np.sum(self.w ** 2)   
                
            self.cost_history.append(cost)
        
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        assert self.w is not None and self.b is not None, "Call fit() first."
        return (self.predict_proba(X) >= threshold).astype(int)