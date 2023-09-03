import numpy as np
import pandas as pd


class MyLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, x_dt, y_dt):
        # Calculating the mean of x and y
        x_mean = np.mean(x_dt)
        y_mean = np.mean(y_dt)

        # Initializing the numerator and denominator for slope calculation
        numerator = 0
        denominator = 0

        for i in range(len(x_dt)):
            numerator += (x_dt[i] - x_mean) * (y_dt[i] - y_mean)
            denominator += (x_dt[i] - x_mean) ** 2

        # Calculate the slope (m) and intercept (b)
        self.m = numerator / denominator
        self.b = y_mean - self.m * x_mean

    def predict(self, X_test):
        if self.m is None or self.b is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")

        # Make predictions using the trained model
        predictions = [self.m * x + self.b for x in X_test]
        return predictions
