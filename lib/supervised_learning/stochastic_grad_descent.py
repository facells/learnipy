def apply_sgd(target, x_train, y_train, x_test):
    from sklearn import linear_model

    if target == 'c':
        model = linear_model.SGDClassifier(shuffle=False)
        print('apply stochastic gradient descent classification (on normalized space) \ntheory: https://en.wikipedia.org/wiki/Stochastic_gradient_descent')
    if target == 'r':
        model = linear_model.SGDRegressor(shuffle=False)
        print('apply sochastic gradient descent regression (on normalized space) \ntheory: https://en.wikipedia.org/wiki/Stochastic_gradient_descent')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model