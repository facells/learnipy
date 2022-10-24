def apply_fn(x_):
    from sklearn import preprocessing
    import pandas as PD

    x_scaled = preprocessing.MinMaxScaler().fit_transform(x_.values)
    x_ = PD.DataFrame(x_scaled, columns=x_.columns)
    print('apply feature normalization')  # normalization by column

    return x_