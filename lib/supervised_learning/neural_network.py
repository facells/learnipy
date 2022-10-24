def apply_nn(target, x_train, y_train, x_test, y_test, complexity, feat, nclass, o, sparseness):
    import math
    import re

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()  # turn dataframe to numpy
    print('apply deep learning neural networks\ntheory: https://en.wikipedia.org/wiki/Deep_learning')
    # l2=2
    # nu=5
    nl = int((complexity / 2) * 10)
    nu = int(math.sqrt(feat * nclass))  # automatic selection of num. layers and nodes

    r_ = re.findall(r'-s.nn=(.+?) ', o)
    if len(r_[0]) > 1:
        nt = r_[0][0]
        nu = int(r_[0][1])
        nl = int(r_[0][2])
        print('using neural network options')
    else:
        print('no neural network options given. detecting best settings:')

    # nu=2**nu
    # l2=0.1**l2 #compute optional values

    # print(x_train)
    # print(y_train)
    # print(x_train.shape)

    if sparseness == 1:  # if data is sparse
        opt = 'adadelta'
    else:  # in normal conditions
        opt = 'adam'
    if target == 'c':  # if classification,
        metric = 'accuracy'
        outactiv = 'sigmoid'
        los = 'sparse_categorical_crossentropy'  # compute classes and options for deep learning
    else:  # if regression
        outactiv = 'linear'
        metric = 'mae'
        los = 'mse'
    if 'nn=l' in o or 'nn=b' in o or 'nn=g' in o or 'nn=r' in o:
        activ = 'tanh'
    else:
        activ = 'selu'

    print(f"layer cofig: actv={activ}, nodes={nu}")
    print(f"network config: optimizer={opt}, out_actv={outactiv}")
    print(f"validator config: loss={los}, metric={metric},")
    # optimizers: adam(robust) sgd(fast) rmsprop(variable) adadelta(for sparse data)
    # activations: linear(lin,-inf/+inf) gelu(nonlin,-0.17/+inf) selu(nonlin,-a/+inf) sigmoid(nonlin,0/1) tanh(nonlin,-1/+1) softplus(nonlin,0/+inf) softsign(nonlin,-1/+1). linear=regression, relu|selu|gelu=general purpose, sigmoid|tanh=binary classification, softplus|softsign=multiclass classifiction
    # losses: hinge kld cosine_similarity mse msle huber binary_crossentropy sparse_categorical_crossentropy
    # metrics: mape mae accuracy top_k_categorical_accuracy categorical_accuracy


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