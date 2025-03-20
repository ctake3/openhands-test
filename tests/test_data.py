import unittest
from unittest.mock import patch
import pandas as pd
from src.data import get_data

class TestData(unittest.TestCase):

    @patch('src.data.yf.download')
    def test_get_data_success(self, mock_download):
        # Create a mock DataFrame
        mock_data = pd.DataFrame({
            'Open': [150, 152],
            'High': [155, 153],
            'Low': [148, 150],
            'Close': [154, 152],
            'Volume': [1000, 1200]
        })
        mock_download.return_value = mock_data

        # Call the function
        get_data()

        # Assert that yf.download was called with the correct arguments
        mock_download.assert_called_once_with("AAPL", start="2023-01-01", end="2023-01-10")

    @patch('src.data.yf.download')
    def test_get_data_failure(self, mock_download):
        # Set up the mock to raise an exception
        mock_download.side_effect = Exception("Test Exception")

        # Call the function and assert that it prints the error message
        import logging
        with self.assertLogs(level='ERROR') as cm:
            get_data()
        self.assertIn("ERROR:root:Error fetching data: Test Exception", cm.output)

if __name__ == '__main__':
    unittest.main()
