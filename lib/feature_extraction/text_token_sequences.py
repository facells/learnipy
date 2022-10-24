def tts(t_, o):
    import pandas as PD
    import re
    import tensorflow as TF

    orig_t_ = t_
    x_ = PD.DataFrame()
    print(
        'tabular data dropped to prevent mixing tabular data and sequence text')  # empty x_ data to avoid mixing text and tabular data
    r_ = re.findall(r'-x.ts=(.+?) ', o)
    wu = int(r_[0])
    t = TF.keras.preprocessing.text.Tokenizer(num_words=1000, lower=True, char_level=False, oov_token=0)
    t.fit_on_texts(t_)
    seq = t.texts_to_sequences(t_)
    seq = TF.keras.preprocessing.sequence.pad_sequences(seq, maxlen=wu)
    print('word indexes:\n', t.index_word)
    t_ = PD.DataFrame(seq)
    # print(t_[0])
    # t_=TF.one_hot(seq,wu)
    # print('one hot seq:', t_)
    # dshape=(t_.shape[1],t_.shape[2])
    # t_=t_.numpy()
    # ft_=[]
    # [ft_.append(i.flatten()) for i in t_]
    # t_=PD.DataFrame(ft_)
    fx = 1
    print('token index sequences:\n', t_) if '-d.data' in o else print(
        f'extracting {wu} token indices sequence features from text')

    return orig_t_, t_, fx, x_