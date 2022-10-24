def __get_bert__(t_, o, batch_size, encoder):
    import tensorflow_text as text
    import tensorflow as TF
    import math
    import tensorflow_hub as TH
    from tqdm import tqdm
    import os
    import pandas as PD

    os.system('pip install tensorflow-text')
    fx = 1

    text_input = TF.keras.layers.Input(shape=(), dtype=TF.string)
    preprocessor = TH.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    del (preprocessor)
    del (encoder)
    pooled_output = outputs["pooled_output"]
    sequence_output = outputs["sequence_output"]
    embedding_model = TF.keras.Model(text_input, [pooled_output, encoder_inputs])
    df = None
    for batch in tqdm(range(math.ceil(len(t_) / batch_size))):
        sentences = TF.constant(t_[batch * batch_size:(batch + 1) * batch_size])
        bert, enc_inps = embedding_model(sentences)
        if df is None:
            df = PD.DataFrame(bert.numpy())
        else:
            df = df.append(PD.DataFrame(bert.numpy()))
    orig_t_ = t_
    t_ = df.reset_index(drop=True)
    print('sync dense bert matrix:\n', t_) if '-d.data' in o else print(f'extracted 768 features')
    print('theory: https://en.wikipedia.org/wiki/BERT_(language_model)')

    return orig_t_, t_, fx

def get_bert(t_, o, batch_size=32):
    import tensorflow_hub as TH
    print(f'extracting 768 features with bert multilanguage cased')

    encoder = TH.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable=True)
    orig_t_, t_, fx = __get_bert__(t_, o, batch_size, encoder)
    return orig_t_, t_, fx

def get_mobert(t_, o, batch_size=32):
    import tensorflow_hub as TH
    print(f'extracting 512 features with mobilebert multilanguage cased')

    encoder = TH.KerasLayer(
        "https://hub.tensorflow.google.cn/tensorflow/mobilebert_multi_cased_L-24_H-128_B-512_A-4_F-4_OPT/1",
        trainable=True)
    orig_t_, t_, fx = __get_bert__(t_, o, batch_size, encoder)
    return orig_t_, t_, fx
