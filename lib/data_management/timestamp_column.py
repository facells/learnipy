def get_timestamp(o):
    import re

    r_ = re.findall(r'-d.ts=(.+?) ', o)
    tscol = (r_[0])
    print(f"using '{tscol}' as timestamp column")
    return tscol