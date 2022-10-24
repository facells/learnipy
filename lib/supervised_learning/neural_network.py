def feedforward(feat, activ, nl, nu, nclass, outactiv):
    import tensorflow as TF

    model = TF.keras.Sequential()
    model.add(TF.keras.layers.Dense(feat, activation=activ))  # initial nodes are=num features
    model.add(TF.keras.layers.Dense(nu * 2, activation=activ))
    [model.add(TF.keras.layers.Dense(nu, activation=activ)) for n in range(0, nl)]
    model.add(TF.keras.layers.Dense(int(nu / 2), activation=activ))
    model.add(TF.keras.layers.Dense(nclass, activation=outactiv))  # output nodes are=nclass
    return model


def imbalancenet(feat, activ, nl, nu, nclass, outactiv):
    import tensorflow as TF

    model = TF.keras.Sequential()
    model.add(TF.keras.layers.Dense(feat, activation=activ))  # initial nodes are=num features
    model.add(TF.keras.layers.Dense(nu * 2, activation=activ))
    model.add(TF.keras.layers.Dropout(0.3))
    [model.add(TF.keras.layers.Dense(nu * 2, activation=activ)) for n in range(0, nl)]
    model.add(TF.keras.layers.Dropout(0.3))
    model.add(TF.keras.layers.Dense(nclass, activation=outactiv))  # output nodes are=nclass
    return model


def lstm(feat, activ, nl, nu, nclass, outactiv, maxval):
    import tensorflow as TF

    model = TF.keras.Sequential()
    model.add(TF.keras.layers.Embedding(maxval, 32, input_length=feat))
    [model.add(TF.keras.layers.LSTM(nu, activation=activ, return_sequences=True)) for n in range(0, nl)]
    model.add(TF.keras.layers.LSTM(nu, activation=activ))
    model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
    return model


def blstm(feat, activ, nl, nu, nclass, outactiv, maxval):
    import tensorflow as TF

    model = TF.keras.Sequential()
    model.add(TF.keras.layers.Embedding(maxval, nu, input_length=feat))
    [model.add(
        TF.keras.layers.Bidirectional(TF.keras.layers.LSTM(int(nu / 2), activation=activ, return_sequences=True))) for n
     in range(0, nl)]
    model.add(TF.keras.layers.Bidirectional(TF.keras.layers.LSTM(int(nu / 2), activation=activ)))
    model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
    return model


def gru(feat, activ, nl, nu, nclass, outactiv, maxval):
    import tensorflow as TF

    model = TF.keras.Sequential()
    model.add(TF.keras.layers.Embedding(maxval, nu, input_length=feat))
    [model.add(TF.keras.layers.GRU(nu * 2, activation=activ, return_sequences=True)) for n in range(0, nl)]
    model.add(TF.keras.layers.GRU(nu, activation=activ))
    model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
    return model


def rnn(feat, activ, nl, nu, nclass, outactiv, maxval):
    import tensorflow as TF

    model = TF.keras.Sequential()
    model.add(TF.keras.layers.Embedding(maxval, nu, input_length=feat))
    [model.add(TF.keras.layers.SimpleRNN(nu, activation=activ, return_sequences=True)) for n in range(0, nl)]
    model.add(TF.keras.layers.SimpleRNN(nu, activation=activ))
    model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
    return model


def cnn(dshape, feat, activ, nl, nu, nclass, outactiv, maxval, x_train, x_test, xtrain_inst, xtest_inst):
    import tensorflow as TF

    if len(dshape) == 1:  # for tables
        model = TF.keras.Sequential()
        model.add(TF.keras.layers.Embedding(maxval, nu, input_length=feat))
        [model.add(TF.keras.layers.Conv1D(int(nu / 2), 7, activation=activ, padding='same')) for n in range(0, nl)]
        # model.add(TF.keras.layers.Conv1D(32, 7, activation=activ,padding='same'))
        model.add(TF.keras.layers.GlobalMaxPooling1D())
        model.add(TF.keras.layers.Dropout(0.3))
        model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
    if len(dshape) == 2:  # for images
        x_train = x_train.reshape((xtrain_inst, dshape[0], dshape[1], 3))
        x_test = x_test.reshape((xtest_inst, dshape[0], dshape[1], 3))
        model = TF.keras.Sequential()
        [model.add(
            TF.keras.layers.Conv2D(nu, kernel_size=(3, 3), activation=activ, input_shape=(dshape[0], dshape[1], 3))) for
         n in range(0, nl)]
        model.add(TF.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(TF.keras.layers.Conv2D(nu * 2, kernel_size=(3, 3), activation=activ))
        model.add(TF.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(TF.keras.layers.Flatten())
        model.add(TF.keras.layers.Dropout(0.5))
        model.add(TF.keras.layers.Dense(nclass, activation=outactiv))

    return x_train, x_test, model