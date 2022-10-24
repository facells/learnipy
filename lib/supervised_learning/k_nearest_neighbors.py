def apply_knn(target, x_train, y_train, x_test):
    from sklearn import neighbors

    if target == 'c':
        model = neighbors.KNeighborsClassifier()
        print('apply k nearest neighbors classification \ntheory: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm')
    if target == 'r':
        model = neighbors.KNeighborsRegressor()
        print('apply k nearest neighbors regression \ntheory: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model