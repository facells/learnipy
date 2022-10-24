def apply_r(target, x_train, y_train, x_test):
    from sklearn import linear_model

    if target == 'c':
        model = linear_model.LogisticRegression(max_iter=5000)
        print('apply logistic regression classification \ntheory:https://en.wikipedia.org/wiki/Logistic_regression')
    if target == 'r':
        model = linear_model.LinearRegression()
        print('apply linear regression \ntheory: https://en.wikipedia.org/wiki/Linear_regression')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model