def prepare_prediction_h4(f, x2_, f2, tgtcol, txtcol, tscol):
    import joblib

    loadmodel = joblib.load(f)
    o = f.replace('-', ' -')
    x_ = x2_
    y_, t_, d_ = None, None, None

    print(f"apply {o} to {f2}")
    # print(tgtcol) #use model filename as o and test set as the main dataset to go into the pipeline with the same settings as the model trained
    if 'tgtcol' in locals() and tgtcol in x_.columns:
        y_ = x_[tgtcol]
        x_ = x_.drop(columns=[tgtcol])
        task = 's'
        print('target found')
    if 'txtcol' in locals() and txtcol in x_.columns:
        t_ = x_[txtcol]
        x_ = x_.drop(columns=[txtcol])
        print('text found')
    if 'tscol' in locals() and tscol in x_.columns:
        d_ = x_[tscol]
        x_ = x_.drop(columns=[tscol])
        print('date found')

    return x_, y_, t_, d_


def prepare_prediction_h5(f, x2_, f2, tgtcol, txtcol, tscol, datatype, names_):
    import tensorflow as TF
    import pandas as PD

    loadmodel = TF.keras.models.load_model(f)
    o = f.replace('-', ' -')
    x_ = x2_
    y_, t_, d_, task = None, None, None,""
    print(
        f"apply {o} to {f2}")  # use model filename as o and test set as the main dataset to go into the pipeline with the same settings as the model trained
    if datatype == 'zip':
        x2_ = PD.DataFrame(names_)
    if 'tgtcol' in locals() and tgtcol in x_.columns:
        y_ = x_[tgtcol]
        task = 's'
        print('target found')
        if datatype == 'csv':
            x_ = x_.drop(columns=[tgtcol])
    if 'txtcol' in locals() and txtcol in x_.columns:
        t_ = x_[txtcol]
        x_ = x_.drop(columns=[txtcol])
        print('text found')
    if 'tscol' in locals() and tscol in x_.columns:
        d_ = x_[tscol]
        x_ = x_.drop(columns=[tscol])
        print('date found')

    return x_, y_, t_, d_, task

