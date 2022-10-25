def extract_using_model(x_, y_, imgmodel, image, preprocess_input, d, label, num_i, i_, shape):
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as NP

    b_ = []
    bl_ = []
    d_ = imread(d)
    d_ = resize(d_, (224, 224, 3), anti_aliasing=True)
    x = image.img_to_array(d_)
    b_.append(x)
    bl_.append(label)
    if len(b_) == 16 or num_i == len(i_) - 1:
        x = NP.array(b_)
        x = preprocess_input(x)
        d_ = imgmodel.predict(x)
        dshape = (shape,)
        d_ = d_.reshape((len(b_), shape))
        for id_ in range(len(b_)):
            x_.append(d_[id_])
            y_.append(bl_[id_])

    return x_, y_, dshape