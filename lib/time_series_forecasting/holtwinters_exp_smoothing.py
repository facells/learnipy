def apply_hwes(y_train, y_test):
    import statsmodels.api as SM

    model = SM.tsa.Holt(y_train).fit()
    y_pred = model.predict(len(y_train), len(y_train) + len(y_test) - 1)  # make prediction
    y_pred = y_pred.to_numpy()
    print(
        f"using HoltWinters Exponential Smoothing for time series\ntheory https://en.wikipedia.org/wiki/Exponential_smoothing")

    return y_pred, model