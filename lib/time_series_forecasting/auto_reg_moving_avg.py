def apply_arma(y_train, y_test):
    import statsmodels.api as SM

    model = SM.tsa.ARMA(y_train, order=(3, 2, 1)).fit()  # fit model
    y_pred = model.predict(len(y_train), len(y_train) + len(y_test) - 1)  # make prediction
    y_pred = y_pred.to_numpy()
    print(
        f"using AutoReg Moving Average for time series\ntheory https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model")

    return y_pred, model