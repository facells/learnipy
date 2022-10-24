def apply_svm(kern, mi, x_train, y_train, x_test):
    from sklearn import svm
    print('apply support vector machines\ntheory: https://en.wikipedia.org/wiki/Support-vector_machine')

    if kern == 'p':
        model = svm.NuSVR(kernel='poly', degree=mi)
        print(f"using polynomial kernel={mi}\ntheory: https://en.wikipedia.org/wiki/Polynomial_kernel")
    if kern == 'r':
        model = svm.NuSVR(kernel='rbf', nu=mi / 10)
        print(f"using rbf kernel={mi / 10}\ntheory: https://en.wikipedia.org/wiki/Radial_basis_function_kernel")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model