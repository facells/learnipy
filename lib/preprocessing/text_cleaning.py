def apply_tc(t_):
    t_ = t_.str.replace("\W", ' ', regex=True)
    t_ = t_.str.replace(" {2,30}", ' ', regex=True)
    print('cleaning string from nonword characters and multiple spaces')
    return t_