def apply_cl(y_):
    y_ = (y_ - y_.min()) / (y_.max() - y_.min())
    print('apply target numbers 0-1 normalization')
    return y_