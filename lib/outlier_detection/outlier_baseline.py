def get_baseline(target, x_train, y_train, x_test, y_test):
    from sklearn import dummy, metrics

    print(f"dataset outlier detection: training set {x_train.shape}")
    if target == 'c':
        model = dummy.DummyClassifier(strategy='prior')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = metrics.f1_score(y_test, y_pred)
        print(f"baseline outlier detection: F1= {score:.3f}")
    if target == 'r':
        model = dummy.DummyRegressor(strategy='mean')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = metrics.r2_score(y_test, y_pred)
        print(f"baseline outlier detection: R2= {score:.3f}")

    return score