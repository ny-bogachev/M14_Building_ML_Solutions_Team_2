import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error
from preprocess import preprocess

def model_initialization(X, y):
    train_size = int(len(X) * 0.9)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    return X_train, y_train, X_test, y_test

def model_train(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib_file = "models/random_forest_regressor_model.joblib"
    joblib.dump(model, joblib_file)
    print(f"Model saved to {joblib_file}")
    return model

def model_mse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    # Load your dataframe here
    df = pd.read_csv("trips_2024_07_02.csv.csv")
    X, y = preprocess(df)
    X_train, y_train, X_test, y_test = model_initialization(X, y)
    model = model_train(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = model_mse(y_test, y_pred)
    print(f"Model MSE: {mse}")
