def get_person_correlation(x_, y_=None):
    import pandas as PD
    import datetime as DT
    import sys

    if 'y_' is not None:
        x_ = PD.concat([x_, y_], axis=1)
    x_ = PD.get_dummies(x_)
    # x_=x_.reset_index(drop=True)  # get one-hot values and restart row index from 0
    print("correlation matrix on one-hot values:\n" + x_.corr().to_string() + "\n")
    af = open('analysis.txt', 'a')
    af.write("correlation matrix on one-hot values:\n\n" + x_.corr().to_string() + "\n\n")
    af.close()
    print('theory: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient')
    timestamp = DT.datetime.now()
    print(f"-u.corr stops other tasks\ntime:{timestamp}")
    sys.exit()