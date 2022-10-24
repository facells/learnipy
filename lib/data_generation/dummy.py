def generate_normdist_rnd_noise(o):
    import re
    import pandas as PD
    from sklearn import datasets

    # generate data with a normal distribution filled with random noise
    r_ = re.findall(r'-g.d=(.+?) ', o)
    ns = int(r_[0][0])
    nf = int(r_[0][1])
    ni = int(r_[0][2])
    ns = ns * 1000
    nf = nf * 10
    ni = ni * 10
    nr = nf - ni
    print(f'generating dataset with {ns} samples and {nf} features, {ni} informative')
    x_, y_ = datasets.make_classification(n_samples=ns, n_features=nf, n_informative=15, n_redundant=5, random_state=1)
    x_ = PD.DataFrame(x_)
    y_ = PD.DataFrame(y_, columns=["class"])
    g_ = PD.concat([x_, y_], axis=1)
    g_.to_csv('gen.csv', sep=',', encoding='utf-8')  # generate a dummy dataset