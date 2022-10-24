def get_if(x_train, y_train):
    from sklearn import ensemble
    print('apply isolation forest\ntheory: https://en.wikipedia.org/wiki/Isolation_forest')
    iso = ensemble.IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    x_train, y_train = x_train[mask], y_train[mask]  # select all rows that are not outliers
    print(f"dataset after outlier detection: training set {x_train.shape}")

    return x_train, y_train