def apply_mb(target, x_train, y_train, x_test):
    from sklearn import linear_model, dummy

    if target == 'c':
        model = linear_model.LogisticRegression(max_iter=5000)
        print('evaluate anomaly removal with linear models')
    if target == 'r':
        model = dummy.DummyRegressor(strategy='mean')
        print('compute mean baseline')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model