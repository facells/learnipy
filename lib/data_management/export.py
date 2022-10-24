def do_export(x_, y_, o):
    import re
    import pandas as PD

    r_ = re.findall(r'-d.export=(.+?) ', o)
    xfn = r_[0]
    y_ = y_.rename('class')
    n_ = PD.concat([x_, y_], axis=1)
    print('exporting processed dataset')
    af = open(f"{xfn}", 'w')
    af.write(n_.to_csv())
    af.close()
    print(f"data saved as {xfn}")