def apply_lcm(target, x_train, y_train, x_test):
    from sklearn import discriminant_analysis, cross_decomposition

    if target == 'c':
        model = discriminant_analysis.LinearDiscriminantAnalysis()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print('apply LinearDiscriminantAnalysis classification \ntheory: https://en.wikipedia.org/wiki/Linear_discriminant_analysis')
    if target == 'r':
        model = cross_decomposition.PLSRegression(max_iter=500)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = y_pred.flatten()
        print('apply PartialLeastSquare regression \ntheory: https://en.wikipedia.org/wiki/Partial_least_squares_regression')

    return y_pred, model