def custom_dict(t_, o):
    import re
    import urllib
    from tqdm import tqdm
    import pandas as PD
    fx = 1
    orig_t_ = t_  # keep text for wordcloud
    r_ = re.findall(r'-x.d=(.+?) ', o)
    l = r_[0]
    print('extracting features from dictionary')
    if l == 'p':
        print('using psy.dic')
        l_ = urllib.request.urlopen(
            'https://raw.githubusercontent.com/facells/learnipy/main/resources/psy.dic').read().decode(
            'utf-8').splitlines()
    elif l == 'e':
        print('using emo.dic')
        l_ = urllib.request.urlopen(
            'https://raw.githubusercontent.com/facells/learnipy/main/resources/emo.dic').read().decode(
            'utf-8').splitlines()
    elif l == 'd':
        print('using dom.dic')
        l_ = urllib.request.urlopen(
            'https://raw.githubusercontent.com/facells/learnipy/main/resources/dom.dic').read().decode(
            'utf-8').splitlines()
    else:
        print('using custom dictionary')
        l_ = open(l, encoding="utf8").read().splitlines()
    h = l_.pop(0)
    h_ = h.split(',')
    t2_ = []
    hf = len(h_)
    # print(dir())
    print(f"dimensions: {h}")
    for t in tqdm(t_):
        t = ' ' + t + ' '
        c = t.count(' ')
        i_ = [0 for i_ in range(hf)]
        for i in l_:
            w_ = i.split(',')
            w = w_.pop(0)
            if re.search(w, t):
                for v in w_:
                    i_[int(v)] = i_[int(v)] + 1
        i_ = [j / c for j in i_]
        t2_.append(i_)
    t_ = PD.DataFrame(t2_, columns=h_)
    del (t2_)
    print('sync sparse matrix from lexicon:\n', t_) if '-d.data' in o else print(f'extracted {hf} features with {l}')

    return orig_t_, t_, fx