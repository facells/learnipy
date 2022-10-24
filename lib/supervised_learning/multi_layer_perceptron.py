def apply_mlp(target, x_train, y_train, x_test):
    from sklearn import neural_network

    if target == 'c':
        model = neural_network.MLPClassifier(random_state=1)
        print('apply multi layer perceptron classification \ntheory: https://en.wikipedia.org/wiki/Multilayer_perceptron')
    if target == 'r':
        model = neural_network.MLPRegressor(random_state=1)
        print('apply multi layer perceprtron regression \ntheory: https://en.wikipedia.org/wiki/Multilayer_perceptron')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, model