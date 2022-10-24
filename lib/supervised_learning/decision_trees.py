def apply_dt(target, x_train, y_train, x_test):
    from sklearn import tree

    if target == 'c':
        model = tree.DecisionTreeClassifier()
        print('apply decision trees classification\ntheory https://en.wikipedia.org/wiki/Decision_tree_learning')
    if target == 'r':
        model = tree.DecisionTreeRegressor()
        print('apply decision trees regression\ntheory https://en.wikipedia.org/wiki/Decision_tree_learning')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model