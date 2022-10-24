def apply_arima(y_train, y_test):
    import statsmodels.api as SM

    model = SM.tsa.ARIMA(y_train, order=(3, 2, 1)).fit()  # fit model
    y_pred = model.predict(len(y_train), len(y_train) + len(y_test) - 1, typ='levels')  # make prediction
    y_pred = y_pred.to_numpy()
    print(
        f"using AutoReg Integrated Moving Average for time series\ntheory https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average")
    return y_pred, model