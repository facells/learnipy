def get_lof(x_train, y_train):
    from sklearn import neighbors

    print('apply local outlier factor\ntheory: https://en.wikipedia.org/wiki/Local_outlier_factor')
    lof = neighbors.LocalOutlierFactor()
    yhat = lof.fit_predict(x_train)
    mask = yhat != -1
    x_train, y_train = x_train[mask], y_train[mask]  # select all rows that are not outliers
    print(f"dataset after outlier detection: training set {x_train.shape}")

    return x_train, y_train