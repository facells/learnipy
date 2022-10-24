def apply_sarima(y_train, y_test):
    import statsmodels.api as SM

    model = SM.tsa.SARIMAX(y_train, order=(3, 2, 1)).fit(disp=False)
    y_pred = model.predict(len(y_train), len(y_train) + len(y_test) - 1)  # make prediction
    y_pred = y_pred.to_numpy()
    print(f"using Seasonal AutoReg Integrated Moving Average for time series\ntheory https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average")

    return y_pred, model