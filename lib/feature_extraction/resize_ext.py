def resize_ext(o):
    import re

    r_ = re.findall(r'-x.rsz=(.+?) ', o)
    size = int(r_[0])
    imgfeats = (size * size) * 3
    print(f"using image size {size}x{size}, extract {imgfeats} features")
    return size, imgfeats

def extract(x_, y_, d, size, label):
    from skimage.io import imread
    from skimage.transform import resize

    d_ = imread(d)
    d_ = resize(d_, (size, size, 3), anti_aliasing=True)
    dshape = (size, size)
    d_ = d_.flatten()
    x_.append(d_)
    y_.append(label)  # put in x_ and y_ extracted features and labels

    return x_, y_, dshape