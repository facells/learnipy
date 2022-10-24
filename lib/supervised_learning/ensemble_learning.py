def apply_random_forest(target, x_train, y_train, x_test):
    from sklearn import ensemble

    if target == 'c':
        model = ensemble.RandomForestClassifier(random_state=1)
        print('apply random forest classification\ntheory: https://en.wikipedia.org/wiki/Random_forest')
    if target == 'r':
        model = ensemble.RandomForestRegressor(random_state=1)
        print('apply random forest regression\ntheory: https://en.wikipedia.org/wiki/Random_forest')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model


def apply_adaboost(target, x_train, y_train, x_test):
    from sklearn import ensemble

    if target == 'r':
        model = ensemble.AdaBoostRegressor(random_state=1)
        print('apply adaboost regression\ntheory: https://en.wikipedia.org/wiki/AdaBoost')
    if target == 'c':
        model = ensemble.AdaBoostClassifier(random_state=1)
        print('apply adaboost classification\ntheory: https://en.wikipedia.org/wiki/AdaBoost')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model


def apply_xgboost(target, x_train, y_train, x_test):
    import xgboost

    if target == 'c':
        model = xgboost.XGBClassifier()
        print('apply gradient boosting classification\ntheory: https://en.wikipedia.org/wiki/Gradient_boosting')
    if target == 'r':
        model = xgboost.XGBRegressor()
        print('apply gradient boosting regression\ntheory: https://en.wikipedia.org/wiki/Gradient_boosting')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model