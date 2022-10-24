def apply_ir(x_):
    from sklearn import utils
    x_ = utils.shuffle(x_, random_state=1)
    print('apply instance position randomization')  #
    return x_


