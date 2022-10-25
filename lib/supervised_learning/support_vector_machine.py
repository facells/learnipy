def apply_svm(target, kern, mi, x_train, y_train, x_test):
    from sklearn import svm
    print('apply support vector machines\ntheory: https://en.wikipedia.org/wiki/Support-vector_machine')

    if target == 'c':
      if kern == 'p':
        model = svm.NuSVC(kernel='poly', degree=mi)
        print('using polynomial kernel\ntheory: https://en.wikipedia.org/wiki/Polynomial_kernel')
    if kern == 'r':
        model = svm.NuSVC(kernel='rbf', nu=mi / 10)
    if target == 'r':
      if kern == 'p':
          model = svm.NuSVR(kernel='poly', degree=mi)
          print(f"using polynomial kernel={mi}\ntheory: https://en.wikipedia.org/wiki/Polynomial_kernel")
      if kern == 'r':
          model = svm.NuSVR(kernel='rbf', nu=mi / 10)
          print(f"using rbf kernel={mi / 10}\ntheory: https://en.wikipedia.org/wiki/Radial_basis_function_kernel")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model