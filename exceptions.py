from typing import Optional

import pandas as pd


class StockDataProcessorError(Exception):
    """
    Base exception for Stock Data Processor errors.

    This is a parent class for all custom exceptions in the module,
    allowing for more granular error handling.
    """
    pass


class APIAccessError(StockDataProcessorError):
    """
    Raised when there are issues accessing the Alpha Vantage API.

    This could be due to:
    - Invalid API key
    - Network issues
    - Rate limiting
    - Unexpected API response
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        """
        Initialize the API access error.

        Args:
            message (str): Descriptive error message
            status_code (Optional[int]): HTTP status code, if applicable
        """
        self.status_code = status_code
        super().__init__(f"API Access Error: {message}")


class DataProcessingError(StockDataProcessorError):
    """
    Raised when there are issues processing stock data.

    This could be due to:
    - Invalid data format
    - Unexpected data structure
    - Data type conversion issues
    """

    def __init__(self, message: str, problematic_data: Optional[pd.DataFrame] = None):
        """
        Initialize the data processing error.

        Args:
            message (str): Descriptive error message
            problematic_data (Optional[pd.DataFrame]): DataFrame causing the error
        """
        self.problematic_data = problematic_data
        super().__init__(f"Data Processing Error: {message}")


class StockDataValidationError(Exception):
    pass