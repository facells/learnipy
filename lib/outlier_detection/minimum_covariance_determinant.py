def get_mcd(x_train, y_train):
    from sklearn import covariance
    print(
        'apply minimum covariance determinant\ntheory: https://en.wikipedia.org/wiki/Covariance_'\
        'matrix#Covariance_matrix_as_a_parameter_of_a_distribution')
    ee = covariance.EllipticEnvelope(contamination=0.01)
    yhat = ee.fit_predict(x_train)
    mask = yhat != -1
    x_train, y_train = x_train[mask], y_train[mask]  # select all rows that are not outliers
    print(f"dataset after outlier detection: training set {x_train.shape}")

    return x_train, y_train