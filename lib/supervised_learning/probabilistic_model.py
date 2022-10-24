def apply_pb(target, x_train, y_train, x_test):
    from sklearn import naive_bayes, linear_model

    if target == 'c':
        model = naive_bayes.ComplementNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print('apply complement naive bayes classification (on normalized space)\ntheory: https://en.wikipedia.org/wiki/Naive_Bayes_classifier')
    if target == 'r':
        model = linear_model.BayesianRidge()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = y_pred.flatten()
        print('apply bayesian ridge regression (on normalized space)\ntheory: https://en.wikipedia.org/wiki/Bayesian_linear_regression')

    return y_pred, model