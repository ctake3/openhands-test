import yfinance as yf

def get_data():
    try:
        # Fetch data for Apple (AAPL)
        data = yf.download("AAPL", start="2023-01-01", end="2023-01-10")
        print(data)

    except Exception as e:
        import logging
        logging.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    get_data()
