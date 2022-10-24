def get_string(o):
    import re

    r_ = re.findall(r'-d.s=(.+?) ', o)
    txtcol = (r_[0])
    print(f"using '{txtcol}' as string column")
    return txtcol