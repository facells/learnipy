def apply_ir(x_):
    x_ = x_.sample(frac=1).reset_index(drop=True)
    print('apply instance randomization in the training set')  # shuffle instances
    return x_


def apply_ir_v2(x_):
    from sklearn import utils
    x_ = utils.shuffle(x_, random_state=1)
    print('apply instance position randomization')  #
    return x_


#TODO fanno la stessa cosa? o hanno risultati finali diversi?