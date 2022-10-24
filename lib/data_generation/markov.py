def generate_sentence(f):
    import os
    import sys

    print('generating sentence from training data:')
    os.system('pip install markovify')
    import markovify

    ft = open(f, encoding='utf8').read()
    # text_model = markovify.NewlineText(ft, state_size = 2)  # for poetry (d.headline_text= pandas serie of newline text)
    text_model = markovify.Text(ft, state_size=3)  # for long well punctuated text (f=string of text)
    print(text_model.make_sentence())
    sys.exit()
    # print(text_model.make_short_sentence(280))
