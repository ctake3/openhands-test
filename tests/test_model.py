import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.model import train_model, evaluate_model

class TestModel(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame({
            'High': [150, 155, 160, 165, 170],
            'Low': [140, 145, 150, 155, 160],
            'Open': [145, 150, 155, 160, 165],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'Close': [148, 153, 158, 163, 168]
        })

    def test_train_model(self):
        model, X_test, y_test = train_model(self.data)
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(len(X_test), 1)  # Check test set size (20% of 5 is 1)
        self.assertEqual(len(y_test), 1)

    def test_evaluate_model(self):
        model, X_test, y_test = train_model(self.data)
        mse = evaluate_model(model, X_test, y_test)
        self.assertIsInstance(mse, float)

if __name__ == '__main__':
    unittest.main()
