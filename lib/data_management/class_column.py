def get_class(o):
    import re

    tgtcol = "class"

    if re.search(r'-d.c=[a-zA-Z]', o):
        r_ = re.findall(r'-d.c=(.+?) ', o)
        tgtcol = (r_[0])
        print(f"'{tgtcol}' is the target class column")
    if re.search(r'-d.c=[0-9]', o):
        r_ = re.findall(r'-d.c=(.+?) ', o)
        tgtcol = int(r_[0])
        print(f"'{tgtcol}' is the target class index")

    return tgtcol
