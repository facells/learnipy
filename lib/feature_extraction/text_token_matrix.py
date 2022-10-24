def ttm(t_, o):
    import re
    import tensorflow as TF
    import pandas as PD

    orig_t_ = t_
    r_ = re.findall(r'-x.tm=(.+?) ', o)
    wu = int(r_[0])
    t = TF.keras.preprocessing.text.Tokenizer(num_words=wu, lower=True, char_level=False, oov_token=wu)
    t.fit_on_texts(t_)
    seq = t.texts_to_sequences(t_)
    wv_ = t.sequences_to_matrix(seq, mode='freq')
    t_ = PD.DataFrame(wv_)
    print('word indexes:\n', t.index_word)
    fx = 1
    print('token freq matrix:\n', t_) if '-d.data' in o else print(
        f'extracting {wu} token frequence features from text')

    return orig_t_, t_, fx