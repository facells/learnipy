def get_drop_column(o):
    import re
    r_ = re.findall(r'-d.x=(.+?) ', o)
    drop = r_[0].split(',')
    return drop