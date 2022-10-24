def apply_trs(t_, stopw=r'\b.{1,3}\b'):
    t_ = t_.str.replace(stopw, ' ')
    t_ = t_.str.replace(r'\s+', ' ')
    return t_