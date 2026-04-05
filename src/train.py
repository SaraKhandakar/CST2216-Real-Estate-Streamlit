from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_models(x_train, y_train, x_test, y_test):
    results = {}

    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    results["Linear Regression"] = {"model": lr_model, "mae": lr_mae}

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    results["Random Forest"] = {"model": rf_model, "mae": rf_mae}

    return results