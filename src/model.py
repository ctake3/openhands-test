import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf

def train_model(data):
    # Drop rows with missing values
    data = data.dropna()

    # Prepare data for linear regression
    X = data[['High', 'Low', 'Open', 'Volume']]
    y = data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse

def backtest(data):
    # Simple walk-forward backtesting
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:]

    X_train = train[['High', 'Low', 'Open', 'Volume']]
    y_train = train['Close']

    X_test = test[['High', 'Low', 'Open', 'Volume']]
    y_test = test['Close']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Backtesting Mean Squared Error: {mse}")

def predict_price(model, features):
    # Predict the closing price for a given set of features
    features_df = pd.DataFrame([features])  # Convert to DataFrame
    predicted_price = model.predict(features_df)[0]
    print(f"Predicted Closing Price: {predicted_price}")
return predicted_price
return predicted_price

if __name__ == "__main__":
    # Fetch data
    data = yf.download("AAPL", start="2023-01-01", end="2023-03-01")
    print(data.columns)
    data = data.rename(columns={col: "".join(col) for col in data.columns})

    # Train the model
    model, X_test, y_test = train_model(data)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Backtesting
    backtest(data)

    # Prediction
    predict_price(model, data.iloc[-1][['High', 'Low', 'Open', 'Volume']])
