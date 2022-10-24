# TO DO: add drop features by index, gen markov chains, gen gan

import warnings


warnings.filterwarnings('ignore')
import datetime as DT
import pandas as PD
import numpy as NP
import tensorflow as TF
import zipfile as ZF
from sklearn import preprocessing, model_selection, tree, metrics, decomposition

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as MP

MP.rcParams["figure.figsize"] = (5, 4)
import nltk

nltk.download('punkt', quiet=True)
import sys
import re
import os
import math
import joblib
from tqdm import tqdm

NP.random.seed(1)
TF.random.set_seed(2)
TF.compat.v1.logging.set_verbosity(TF.compat.v1.logging.ERROR)
MP.clf()

print("\n\n")
print('---START PROCESS---')
timestamp = DT.datetime.now()
print(f"time:{timestamp}")

# ---system option reading
o = sys.argv[1] + ' '

if '-h' in o:
    if os.path.exists("README.md"):
        with open("README.md", "r") as f:
            print(f.read())
    sys.exit()

# import image pretrained models
if '-x.resnet' in o:
    from lib.feature_extraction import resnet

    imgmodel, image, preprocess_input = resnet.get_resnet50()

if '-x.vgg' in o:
    from lib.feature_extraction import vgg

    imgmodel, image, preprocess_input = vgg.get_vgg16()

if '-x.effnet' in o:
    from lib.feature_extraction import effnet

    imgmodel, image, preprocess_input = effnet.get_effnetb2()

# ---files reading
try:
    f = sys.argv[2]  # take dataset or model
except:
    print('no file loaded')
    sys.exit()

try:
    f2 = sys.argv[3]  # take testset or newdataset
except:
    print('using training set')

# ---generate data
if '-g.d=' in o:  # generate data with a normal distribution filled with random noise
    from lib.data_generation import dummy

    dummy.generate_normdist_rnd_noise(o)

if '-g.mct' in o:  # markov chains to generate text with next word probability
    from lib.data_generation import markov

    markov.generate_sentence(f)


datatype = ''

# ---initialize default settings
if 'f2' in locals() and '.csv' in f2:  # import csv test set
    datatype = 'csv'

if 'f2' in locals() and '.zip' in f2:
    datatype = 'zip'

if '-x.rsz' in o:  # extract features from image with resize
    from lib.feature_extraction import resize_ext

    size, imgfeats = resize_ext.resize_ext(o)

else:  # otherwise apply default size 28x28
    size = 16
    imgfeats = (size * size) * 3
    # print(f"using default image size {size}x{size}, extract {imgfeats} features")

if '-d.ts=' in o:  # get name of timestamp column
    from lib.data_management import timestamp_column

    tscol = timestamp_column.get_timestamp(o)

else:  # otherwise apply default name
    tscol = 'date'
    print('extract "date" as timestamp column')

if '-d.s=' in o:  # get name of string column
    from lib.data_management import string_column

    txtcol = string_column.get_string(o)
else:  # otherwise apply default name
    txtcol = 'text'
    print('extract "text" as string column')

if '-d.c=' in o:  # get name of class column
    from lib.data_management import class_column

    tgtcol = class_column.get_class(o)

else:  # otherwise apply default name
    if '.csv' in f or datatype == 'csv':
        tgtcol = 'class'
        print('extract "class" as target class')
    elif '.zip' in f or datatype == 'zip':
        tgtcol = 1
        print('extract index 1 from comma separated file name as target class')
    else:
        ###TODO eccezione

if '-d.x=' in o:  # get name of column to drop
    from lib.data_management import drop_column

    drop = drop_column.get_drop_column(o)

if '-d.t=c' in o:  # auto define classification or regression
    target = 'c'
    print('task: classification')
if '-d.t=r' in o:
    target = 'r'
    print('task: regression')

# ---reading test set
if 'f2' in locals():  # import csv test set
    if '.csv' in f2:
        testname = f2.replace('.csv', '')
        datatype = 'csv'
        x2_ = PD.read_csv(f2, sep=',', encoding='utf8')
        testinst = len(x2_.index)
        print('using test set or new set')
    if '.zip' in f2:
        testname = f2.replace('.zip', '')
        datatype = 'zip'
        zip = ZF.ZipFile(f2, 'r')
        i_ = zip.namelist()
        print('using test set or new set')
        if ',' in i_[0]:  # extract supervised data from files in zip
            x2_ = []
            y2_ = []
            names_ = []
            for i in i_:
                l_ = i.split(',')
                if l_[1] != '':
                    label = l_[1]
                else:
                    label = 0
                d = zip.open(i)
                # print(d)
                if '.jpg' in i or '.png' in i:
                    d_ = imread(d)
                    names_.append(d.name)
                    d_ = resize(d_, (size, size, 3), anti_aliasing=True)
                    dshape = (size, size)
                    d_ = d_.flatten()
                    x2_.append(d_)
                    y2_.append(label)  # read, resize and flatten images
            # ADD other file formats extraction
            x2_ = NP.array(x2_).astype('float') / 255.0
            y2_ = NP.array(y2_).astype('int')
            x2_ = PD.DataFrame(x2_)
            y2_ = PD.Series(y2_)
            testinst = len(x2_.index)
            print(testinst)

# ---import saved models
if '.h4' in f and '-d.pred' in o:  # import machine learning saved model

    from lib.data_management import make_prediction

    x_, y_, t_, d_ = make_prediction.prepare_prediction_h4(f, x2_, f2, tgtcol, txtcol, tscol)


if '.h5' in f and '-d.pred' in o:  # import deep learning saved model
    from lib.data_management import make_prediction

    x_, y_, t_, d_, task = make_prediction.prepare_prediction_h5(f, x2_, f2, tgtcol, txtcol, tscol, datatype, names_)

# ---read data
if '.csv' in f:  # import .csv training set or (if there is a test set) create training+test set
    o = o.replace('-', ' -')
    print(f"processing {f} with {o}")
    filename = f.replace('.csv', '')
    datatype = 'csv'
    x_ = PD.read_csv(f, sep=',', encoding='utf8')
    traininst = len(x_.index)
    dshape = [1]
    x_ = x_.reset_index(drop=True)  # start row index from 0
    if '-p.ir' in o:
        from lib.preprocessing import instance_randomization
        x_ = instance_randomization.apply_ir(x_)

    if 'x2_' in locals():
        tts = ((100 / (traininst + testinst)) * testinst) / 100
        print(f'test set percentage={tts}')  # compute tts percentage
        if not x2_.empty:
            x_ = PD.concat([x_, x2_], axis=0, ignore_index=True)

    if 'tgtcol' in locals() and tgtcol in x_.columns:  # remove rows with missing values in class and extract target class dataframe
        if not '-d.m=1' in o:
            x_ = x_.dropna(subset=[tgtcol])
            print(f"remove rows with missing values in {tgtcol}")
        y_ = x_[tgtcol]
        x_ = x_.drop(columns=[tgtcol])
        task = 's'
        print('target found')
    else:
        print('no class given. only unsupervised tasks enabled.\nfor supervised tasks define a target with -d.c=name or name "class" the target column')
        task = 'u'

    if 'txtcol' in locals() and txtcol in x_.columns:  # create dataframe to process text
        t_ = x_[txtcol]
        x_ = x_.drop(columns=[txtcol])
        t_ = t_.reset_index(drop=True)  # start row index from 0
        print(f"taken {txtcol} as string column")

    if 'tscol' in locals() and tscol in x_.columns:  # create dataframe to process text
        d_ = x_[tscol]
        x_ = x_.drop(columns=[tscol])
        d_ = d_.reset_index(drop=True)
        d_ = d_.rename({tscol: 'date'}, axis='columns')  # rename the time column
        print(f"taken {tscol} as date column")

if '.zip' in f:  # extract data from .zip, loading in memory
    o = o.replace('-', ' -')
    print(f"processing {f} with {o}")
    filename = f.replace('.zip', '')
    datatype = 'zip'
    zip = ZF.ZipFile(f, 'r')
    i_ = zip.namelist()
    if ',' in i_[0]:  # extract supervised data from files in zip
        task = 's'
        print('target found, suppose supervised task')
        x_ = []
        y_ = []
        print('reading data from .zip')
        if '-x.resnet' in o:
            print('apply resnet feature extraction')
        elif '-x.vgg' in o:
            print('apply vgg feature extraction')
        elif '-x.effnet' in o:
            print('apply efficientnet feature extraction')
        else:
            print(f'apply {size}x{size} img resize feature extraction')
        for num_i, i in enumerate(tqdm(i_)):
            l_ = i.split(',')
            label = l_[tgtcol]
            d = zip.open(i)
            # print(d)
            if '.jpg' in i or '.png' in i:
                from lib.feature_extraction import extract_features

                if '-x.resnet' in o:  # resnet img feature extraction

                    x_, y_, dshape = extract_features.extract_using_model(imgmodel, image, preprocess_input, d, label, num_i, i_, 2048)

                elif '-x.vgg' in o:  # vgg img feature extraction

                    x_, y_, dshape = extract_features.extract_using_model(imgmodel, image, preprocess_input, d, label, num_i, i_, 512)

                elif '-x.effnet' in o:  # efficientnetB2 img feature extraction

                    x_, y_, dshape = extract_features.extract_using_model(imgmodel, image, preprocess_input, d, label, num_i, i_, 1408)

                else:  # using resize img feature extraction (default)
                    from lib.feature_extraction import resize_ext

                    x_, y_, dshape = resize_ext.extract(d, size, label)

        # ADD other file formats extraction
        x_ = NP.array(x_).astype('float') / 255.0
        x_ = PD.DataFrame(x_)
        traininst = len(x_.index)
        y_ = PD.Series(y_)  # y_=y_.astype('float')

        if 'x2_' in locals() and not '-d.pred' in o:
            tts = ((100 / (traininst + testinst)) * testinst) / 100
            print(f'test set percentage={tts}')  # compute tts percentage
            x_ = PD.concat([x_, x2_], axis=0, ignore_index=True)
            y_ = PD.concat([y_, y2_], axis=0, ignore_index=True)

    else:  # extract unsupervised data from files in zip
        task = 'u'
        print('no class given. only unsupervised tasks enabled.\nfor supervised tasks format your data as name,label,.jpg')
        x_ = []
        for i in i_:
            d = zip.open(i)
            # print(d)
            if '.jpg' in i or '.png' in i:
                d_ = imread(d)
                dshape = d_.shape
                d_ = d_.flatten()
                x_.append(d_)  # read and flatten images
        # ADD other file formats extraction
        x_ = NP.array(x_).astype('float') / 255.0
        x_ = PD.DataFrame(x_)

if not '.zip' in f and not '.csv' in f and not '.zip' in f2 and not '.csv' in f2:
    print('please input a .csv or .zip dataset')
    sys.exit()

# ---drop selected columns
if '-d.x=' in o:
    x_ = x_.drop(columns=drop)

# ---info on visualizations
if '-d.viz' in o:
    print('all scatterplots are 2D-PCA spaces\ntheory: https://en.wikipedia.org/wiki/Principal_component_analysis')

# ---unsupervised learning on unprocessed data

if '-u.arl' in o:  # association rule mining
    from lib.unsupervised_learning import association_rule_learning
    association_rule_learning.arl(x_)

# ---automatically detect target class type
if task == 's':
    if y_.dtype.kind in 'biufc' or '-d.t=r' in o or not '-d.t=c' in o:
        # y_=(y_-y_.min())/(y_.max()-y_.min())
        y_ = y_.astype('float')
        target = 'r'
        print('read target as number, turned to float. to read target as a label use -d.t=c instead')
        if '-p.cn' in o:
            from lib.preprocessing import class_normalization
            y_ = class_normalization.apply_cl(y_)
    else:
        y_ = y_.astype('category').cat.codes.astype('int')
        target = 'c'
        print('read target as label, turned to integer category')

# ---manage filling missing values in features
if 'x_' in locals() and not '-u.arl' in o:
    if x_.isnull().values.any():
        x_ = x_.fillna(0)
        print('filling missing values with 0 by default')  # fill all missing values in x_ with 0

if 'y_' in locals() and '-d.m=1' in o:  # replace missing values with mode or mean
    if y_.isnull().values.any():
        if target == 'r':
            y_ = y_.fillna(y_.mean())
            print('WARNING: there are missing values in the target class, filled with the mean.')  # fill all missing values in y_ with 0
        if target == 'c':
            y_ = y_.fillna(y_.mode())
            print('WARNING: there are missing values in the target class, filled with the mode.')  # fill all missing values in y_ with 0

if 't_' in locals():
    t_ = t_.astype(str)

# ---data preprocessing
if '-p.ir' in o:
    from lib.preprocessing import instance_randomization
    x_ = instance_randomization.apply_ir(x_)

# if '-p.stnd' in o: *TO REMOVE*
#     x_scaled=SK.preprocessing.StandardScaler().fit_transform(x_.values)
#     x_=PD.DataFrame(x_scaled, columns=x_.columns)
#     print('apply feature standardization')

if '-p.tl' in o:
    from lib.preprocessing import text_lowercase
    t_ = text_lowercase.apply_tl(t_)

if '-p.tc' in o:
    from lib.preprocessing import text_cleaning
    t_ = text_cleaning.apply_tc(t_)

if '-p.trs' in o:
    from lib.preprocessing import text_regex_stopword
    t_ = text_regex_stopword.apply_trs(t_)
    print(f'apply regex 1 to 3 lenght word removal')

if '-p.tsw=' in o:
    r_ = re.findall(r'-p.tsw=(.+?) ', o)
    swlist = r_[0]
    swlist = swlist.replace(',', ' | ')
    print(swlist)  # get list of stopwords
    from lib.preprocessing import text_regex_stopword
    t_ = text_regex_stopword.apply_trs(t_, swlist)
    print(f'remove stopwords: {swlist}')

ncols = len(x_._get_numeric_data().columns)
cols = len(x_.columns)  # count of label and numeric columns in dataset

# ---feature reduction (applied in saved models)
if '-r.svd=' in o:  # define dimensions for SVD
    r_ = re.findall(r'-r.svd=(.+?) ', o)

    from lib.feature_reduction import def_dimension
    svdim = def_dimension.def_dimension(r_)

else:
    svdim = int(1 + (cols / 2))
    o = f"-r.svd={svdim}" + o

if '-r.lsa=' in o:  # define dimensions for LSA
    r_ = re.findall(r'-r.lsa=(.+?) ', o)
    from lib.feature_reduction import def_dimension

    lsadim = def_dimension.def_dimension(r_)

    print(f'apply lsadim={lsadim}.')
else:
    lsadim = 50
    o = f"-r.lsa={lsadim}" + o

# ---feature extraction
fx = 0  # define flag for feature extraction

# one-hot
if not x_.empty and not 'zip' in datatype and cols >= ncols:  # if data not empty, .csv and with label columns then extract features from labels, apply SVD
    x_ = PD.get_dummies(x_)
    x_ = x_.reset_index(drop=True)  # get one-hot values and restart row index from 0
    print('async sparse one-hot matrix from labels:\n', x_) if '-d.data' in o else print('apply one-hot binarization of labels by default, obtain sparse async matrix')
    # print(x_.describe())
    # SVD feature reduction
    if not '-d.r=0' in o:  # check whether to run feature reduction or leave data as it is
        if len(x_.columns) > 2:
            from lib.feature_reduction import singular_value_decomposition
            x_ = singular_value_decomposition.apply_svd(svdim, x_, o)

        else:
            x_ = x_
            print('tabular data is small, SVD not applied')

if 't_' in locals() and '-x.' in o:  # extract features from text, apply LSA
    print('original text data:\n', t_) if '-d.data' in o else 0

    if '-x.tm=' in o:  # one hot token matrix
        from lib.feature_extraction import text_token_matrix
        orig_t_, t_, fx = text_token_matrix.ttm(t_, o)

    if '-x.cm=' in o:  # one hot char matrix
        from lib.feature_extraction import text_char_matrix
        orig_t_, t_, fx = text_char_matrix.tcm(t_, o)

    if '-x.ts=' in o:  # one hot sequence matrix
        from lib.feature_extraction import text_token_sequences
        orig_t_, t_, fx, x_ = text_token_sequences.tts(t_, o)

    if '-x.ng=' in o:
        from lib.feature_extraction import ngrams
        orig_t_, t_, fx = ngrams.ng(t_, o, lsadim)

    if '-x.d2v=' in o:  # doc2vec #TODO controllare il seed
        from lib.feature_extraction import doc2vec
        orig_t_, t_, fx = doc2vec.get_doc2vec(t_, o)

    if '-x.bert ' in o:  # bert uncased multi language (contributor: Cristiano Casadei)
        from lib.feature_extraction import bert
        orig_t_, t_, fx = bert.get_bert(t_, o)

    if '-x.mobert ' in o:  # bert uncased multi language (contributor: Cristiano Casadei)
        from lib.feature_extraction import bert
        orig_t_, t_, fx = bert.get_mobert(t_, o)

    if '-x.d=' in o:  # user defined lexical resources
        from lib.feature_extraction import extract_custom_dict
        orig_t_, t_, fx = extract_custom_dict.custom_dict(t_, o)

    # ---data aggregation
    if fx == 1:  # if feature extraction performed concat x_ and t_, else drop t_
        x_ = PD.concat([x_, t_], axis=1)
    else:
        print('no text feature extraction. text column dropped')
        if x_.empty:
            print('no features. prcess stopped')
            sys.exit()

inst = len(x_.index)
feat = len(x_.columns)
print(f'dataset shape: {inst} instances, {feat} features')

# ---class statistics and correlation complexity
if not '-u.corr' in o:
    if '-d.viz' in o and not '-t.' in o:
        if task == 's':
            MP.hist(y_, color='black', edgecolor='black', linewidth=0)
            MP.ylabel('frequency')
            MP.title('class distribution')
            MP.savefig(fname='class-dist')
            MP.show()
            MP.clf()  # class dist
    if target == 'r':
        mu = y_.mean()
        mi = y_.min()
        ma = y_.max()
        sd = y_.std()
        me = y_.median()
        print(f"min={mi:.1f} max={ma:.1f} avg={mu:.3f} sd={sd:.3f} med={me:.3f} numeric target distribution")
        nclass = 1
    if target == 'c':
        print(f"class freq")
        print(y_.value_counts())
        ynp_ = y_.to_numpy()
        classes, counts = NP.unique(ynp_, return_counts=True)
        nclass = len(classes)
        print(f"num classes= {nclass}")

    xc_ = PD.concat([y_, x_], axis=1)
    xcc = xc_.corr()
    xcc = PD.Series(xcc.iloc[0])
    xcc = xcc.iloc[1:]
    complexity = 1 - (xcc.abs().max())
    print(f"corr. complexity= {complexity:.3f}")  # compute correlation complexity

# ---processed features unsupervised learning: corr, w2v and clustering

if '-u.corr' in o:  # correlation analysis
    from lib.unsupervised_learning import correlation_analysis
    if 'y_' in locals():
        correlation_analysis.get_person_correlation(x_, y_)
    correlation_analysis.get_person_correlation(x_)

if '-u.w2v' in o and 't_' in locals():  # word2vec
    from lib.unsupervised_learning import word2vec
    word2vec.get_word2vec(t_, o)

if '-u.km=' in o:  # kmeans clustering
    from lib.unsupervised_learning import centroid_clustering
    x_ = centroid_clustering.kmeans(x_, o)

if '-u.optics' in o:  # optics clustering
    from lib.unsupervised_learning import density_clustering
    x_ = density_clustering.optics(x_, o)

if '-u.msh' in o:  # meanshift clustering
    from lib.unsupervised_learning import density_clustering

    x_ = density_clustering.mean_shift(x_, o)

if '-u.som' in o:  # self organising map clustering (contributor: Fabio Celli)
    from lib.unsupervised_learning import nn_clustering
    x_ = nn_clustering.self_organizing_map(x_, o, feat)

if '-u.ap' in o:  # affinity propagation clustering
    from lib.unsupervised_learning import affinity_propagation_clustering
    x_ = affinity_propagation_clustering.affinity_propagation(x_, o)

to_implement = '''
if '-u.gxm' in o: #gaussian models expectation maximisation
    r_=re.findall(r'-u.gxm=(.+?) ',o)
    nk=int(r_[0])
    clust = SK.mixture.GaussianMixture(n_components=nk).fit(x_)
    l_=PD.DataFrame(clust.predict)
    l_.columns=['expectmax']
    x_=PD.concat([x_,l_], axis=1)
    g_=x_.groupby('expectmax').mean()
    print('applied expectation maximisation clustering. added 1 feature')
    print('theory: https://en.wikipedia.org/wiki/Expectation-maximization_algorithm')
    af= open('analysis.txt', 'a')
    af.write(g_.to_string()+"\n\n")
    af.close()
    ynp_=l_.to_numpy()
    classes, counts=NP.unique(ynp_, return_counts=True)
    nk=len(classes)
    print(f"num clusters= {nk}")
    if '-d.viz' in o:
        pca=SK.decomposition.PCA(2)
        projected=pca.fit_transform(x_)
        MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk))
        MP.xlabel('component 1')
        MP.ylabel('component 2')
        MP.colorbar()
        MP.title('2D PCA data space expectation maximisation clusters')
        MP.savefig(fname='pca-cluster-space.png')
        MP.show()
        MP.clf() # pca space
'''


# ---exporting
if '-d.export' in o:
    from lib.data_management import export
    export.do_export(x_, y_, o)
    # sys.exit()

# ---preprocessing features
if not '-x.ts=' in o or not 't_' in locals():
    maxval = 2
    if '-p.fn' in o or '-s.sgd' in o or '-s.nb' in o or '-s.nn' in o:  # column normalization range 0-1 (required for sgd and nb)
        from lib.preprocessing import feature_normalization
        x_ = feature_normalization.apply_fn(x_)
if '-x.ts=' in o:
    # x_=x_.div(x_.sum(axis=1), axis=0)
    # print('apply instance normalization') #normalize by row
    # x_scaled=SK.preprocessing.MinMaxScaler().fit_transform(x_.values)
    # x_=PD.DataFrame(x_scaled, columns=x_.columns)
    # print('apply feature normalization') #normalization by column
    maxval = 1000
print(f"max value for neural network embedding= {maxval}")


# ---feature summary
x_ = PD.DataFrame(x_.values.astype(NP.float64), columns=x_.columns)
print('turn all features to float numbers.')
# print(x_)
# print(list(x_))
# print(x_.dtypes)
sparseness = 0
sparsemsg = ''
sparse = sum((x_ == 0).astype(int).sum()) / x_.size  # initialize sparseness check
if sparse >= 0.9:
    sparseness = 1
    sparsemsg = 'WARNING: high data sparseness, better to apply feature reduction. '
print(f"sparseness= {sparse:.3f}")

x_.columns = range(x_.shape[1])  # make columns unique

if '-d.fdist' in o:
    # print(x_.dtypes)
    # print(y_.dtypes)
    from lib.data_management import visualization
    visualization.feature_distribution(x_)

viz_deprecated = '''
if '-d.viz' in o:
    if task=='s':
        pca=SK.decomposition.PCA(2)
        projected=pca.fit_transform(x_)
        MP.scatter(projected[:, 0], projected[:, 1], c=y_, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
        MP.xlabel('component 1')
        MP.ylabel('component 2')
        if target=='c':
            MP.colorbar() 
        MP.title('2D PCA data space and classes')
        MP.savefig(fname='pca-data-space.png')
        MP.show()
        MP.clf() #pca space
    else:
        pca=SK.decomposition.PCA(2)
        projected=pca.fit_transform(x_)
        MP.scatter(projected[:, 0], projected[:, 1], c=y_, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', 1))
        MP.xlabel('component 1')
        MP.ylabel('component 2')
        MP.colorbar()
        MP.title('2D PCA unsupervised data space')
        MP.savefig(fname='pca-data-space.png')
        MP.show()
        MP.clf() #pca space
'''

# ---predictions with saved model
if '.h4' in f:  # apply machine learning saved model
    # print(y_)
    # print(len(y_))
    # print(loadmodel.k_ar)
    if '-t.' in o:
        if '-t.a' in o:
            y2_pred = loadmodel.forecast(steps=100)[0]
        else:
            y2_pred = loadmodel.forecast(steps=100)
        y2_pred = PD.DataFrame(y2_pred)
        print(y2_pred)
        n_ = PD.concat([y_, y2_pred], axis=0).reset_index()
        print('applied saved model on supplied dataset. results:\n')
        print(n_)
        af = open(f"{testname}-predictions.csv", 'w')
        af.write(n_.to_csv())
        af.close()
        print(f"data with predictions saved as {testname}-predictions.csv")
        sys.exit()
    else:
        y2_pred = loadmodel.predict(x_)
        y2_pred = PD.DataFrame(y2_pred)
        # print(type())
        n_ = PD.concat([x2_, y2_pred], axis=1)
        print('applied saved model on supplied dataset. results:\n')
        print(n_)
        af = open(f"{testname}-predictions.csv", 'w')
        af.write(n_.to_csv())
        af.close()
        print(f"data with predictions saved as {testname}-predictions.csv")
        sys.exit()

if '.h5' in f:  # apply deep learning saved model
    if '-t.' in o:  # in case of timeseries
        d_ = PD.to_datetime(d_)
        y_ = PD.DataFrame(y_)
        y_ = PD.concat([y_, d_], axis=1)
        y_['date'] = d_
        y_.sort_values(by="date")  # sort x_ in date order
        y_ = y_.drop(columns=['date'])
        y_['date'] = y_.index  # substitute date with integer index
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        y_ = y_.drop(columns=['date'])
        y_ = scaler.fit_transform(y_)
        y_ = NP.reshape(y_, (y_.shape[0], y_.shape[1], 1))
        y2_pred = loadmodel.predict(y_)
        y2_pred = scaler.inverse_transform(y2_pred)
    else:
        if datatype == 'zip':
            x_ = x_.to_numpy()
            print(x_.shape)
            x_ = x_.reshape(-1, dshape[0], dshape[1], 3)
        y2_pred = loadmodel.predict(x_)
    if target == 'c':
        y2_pred = PD.DataFrame(y2_pred)
        y2_pred = y2_pred.idxmax(axis=1)
        # print(y2_pred)
    if target == 'r':
        y2_pred = PD.DataFrame(y2_pred)
    n_ = PD.concat([x2_, y2_pred], axis=1)
    print('applied saved model on supplied test set')
    af = open(f"{testname}-predictions.csv", 'w')
    af.write(n_.to_csv())
    af.close()
    print(f"data with predictions saved as {testname}-predictions.csv")
    sys.exit()

# ---train and test split
# if '-e.cv=' in o: # cross validation *TO REMOVE*
#     r_=re.findall(r'-e.cv=(.+?) ',o)
#     folds=int(r_[0])
#     cv = SK.model_selection.RepeatedKFold(n_splits=folds, random_state=1) # prepare the cross-validation eval

if '-e.tts' in o:
    r_ = re.findall(r'-e.tts=(.+?) ', o)
    split = float(r_[0])
    if 'x2_' in locals():
        split = tts
else:
    split = 0.3
    print('using 70% train 30% test percentage split by default')
if 'split' in locals():  # if split percentage is defined, then split train and test sets
    if '-t.' in o:  # in case of timeseries
        d_ = PD.to_datetime(d_)
        y_ = PD.DataFrame(y_)
        # x_=PD.concat([d_, x_], axis=1)
        # y_=PD.concat([d_, y_], axis=1) #put timestamps to train and test
        x_['date'] = d_
        x_.sort_values(by="date")  # sort x_ in date order
        y_['date'] = d_
        y_.sort_values(by="date")  # sort x_ in date order
        tscol = 'date'
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_, y_, test_size=split, shuffle=False)  # prepare train test eval
    else:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_, y_, test_size=split, shuffle=True, random_state=1)  # prepare train test eval
    xtrain_inst = len(x_train.index)
    feat = len(x_train.columns)
    print(f'training set shape: {xtrain_inst} instances, {feat} features')
    xtest_inst = len(x_test.index)
    feat = len(x_test.columns)
    print(f'test set shape: {xtest_inst} instances, {feat} features')
    x_test2 = x_test  # create a copy of x_test for evaluation in case of shape change

if '-d.data' in o:
    if '-t.' in o:  # in case of timeseries
        print('data ready for learning:')
        print(y_)
    else:
        print('data ready for learning:')
        print(x_)

# ---outlier detection
if '-o.' in o:
    from lib.outlier_detection import outlier_baseline
    score = outlier_baseline.get_baseline(target, x_train, y_train, x_test, y_test)

    # apply algorithms
    if '-o.if' in o:
        from lib.outlier_detection import isolation_forest
        x_train, y_train = isolation_forest.get_if(x_train, y_train)

    if '-o.mcd' in o:
        from lib.outlier_detection import minimum_covariance_determinant
        x_train, y_train = minimum_covariance_determinant.get_mcd(x_train, y_train)

    if '-o.lof' in o:
        from lib.outlier_detection import local_outlier_factor
        x_train, y_train = local_outlier_factor.get_lof(x_train, y_train)

    score = outlier_baseline.get_baseline(target, x_train, y_train, x_test, y_test)

# ---supervised learning
if '-s.base' in o:
    from lib.supervised_learning import majority_baseline
    y_pred, model = majority_baseline.apply_mb(target, x_train, y_train, x_test)

if '-s.nb' in o:
    from lib.supervised_learning import probabilistic_model
    y_pred, model = probabilistic_model.apply_pb(target, x_train, y_train, x_test)

if '-s.lcm' in o:
    from lib.supervised_learning import linear_combination_model
    y_pred, model = linear_combination_model.apply_lcm(target, x_train, y_train, x_test)

if '-s.lr' in o:
    from lib.supervised_learning import regression
    y_pred, model = regression.apply_r(target, x_train, y_train, x_test)

if '-s.sgd' in o:
    from lib.supervised_learning import stochastic_grad_descent
    y_pred, model = stochastic_grad_descent.apply_sgd(target, x_train, y_train, x_test)

if '-s.knn' in o:
    from lib.supervised_learning import k_nearest_neighbors
    y_pred, model = k_nearest_neighbors.apply_knn(target, x_train, y_train, x_test)

if '-s.mlp' in o:
    from lib.supervised_learning import multi_layer_perceptron
    y_pred, model = multi_layer_perceptron.apply_mlp(target, x_train, y_train, x_test)

if '-s.svm' in o and target == 'c':
    if '-s.svm=' in o:
        r_ = re.findall(r'-s.svm=(.+?) ', o)
        kern = r_[0][0]
        mi = int(r_[0][1])
    else:
        kern = 'r'
        mi = 5

    from lib.supervised_learning import support_vector_machine
    y_pred, model = support_vector_machine.apply_svm(kern, mi, x_train, y_train, x_test)

if '-s.rf' in o:
    from lib.supervised_learning import ensemble_learning
    y_pred, model = ensemble_learning.apply_random_forest(target, x_train, y_train, x_test)

if '-s.ada' in o:
    from lib.supervised_learning import ensemble_learning
    y_pred, model = ensemble_learning.apply_adaboost(target, x_train, y_train, x_test)

if '-s.dt' in o:
    from lib.supervised_learning import decision_trees
    y_pred, model = decision_trees.apply_dt(target, x_train, y_train, x_test)

if '-s.xgb' in o:
    from lib.supervised_learning import ensemble_learning
    y_pred, model = ensemble_learning.apply_xgboost(target, x_train, y_train, x_test)

if 'model' in locals() and '-s.' in o:
    if target == 'c':  # compute classes for machine learning
        classes = model.classes_
    if '-d.save' in o and not '-s.nn' in o:  # save machine learning models
        opt = re.sub(r'-d.save|-e\..+|-d.viz| ', '', o)
        if target == 'c':
            opt = '-d.t=c' + opt
        else:
            opt = '-d.t=r' + opt
        joblib.dump(model, f"{filename}{opt}.h4")
        print(f"model saved as {filename}{opt}.h4")
        # af= open(f"{filename}-format4model.csv", 'w')
        # af.write(x_test.to_csv())
        # af.close()

if '-s.nn' in o:
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()  # turn dataframe to numpy
    print('apply deep learning neural networks\ntheory: https://en.wikipedia.org/wiki/Deep_learning')
    # l2=2
    # nu=5
    nl = int((complexity / 2) * 10)
    nu = int(math.sqrt(feat * nclass))  # automatic selection of num. layers and nodes

    r_ = re.findall(r'-s.nn=(.+?) ', o)
    if len(r_[0]) > 1:
        nt = r_[0][0]
        nu = int(r_[0][1])
        nl = int(r_[0][2])
        print('using neural network options')
    else:
        print('no neural network options given. detecting best settings:')

    # nu=2**nu
    # l2=0.1**l2 #compute optional values

    # print(x_train)
    # print(y_train)
    # print(x_train.shape)

    if sparseness == 1:  # if data is sparse
        opt = 'adadelta'
    else:  # in normal conditions
        opt = 'adam'
    if target == 'c':  # if classification,
        metric = 'accuracy'
        outactiv = 'sigmoid'
        los = 'sparse_categorical_crossentropy'  # compute classes and options for deep learning
    else:  # if regression
        outactiv = 'linear'
        metric = 'mae'
        los = 'mse'
    if 'nn=l' in o or 'nn=b' in o or 'nn=g' in o or 'nn=r' in o:
        activ = 'tanh'
    else:
        activ = 'selu'

    print(f"layer cofig: actv={activ}, nodes={nu}")
    print(f"network config: optimizer={opt}, out_actv={outactiv}")
    print(f"validator config: loss={los}, metric={metric},")
    # optimizers: adam(robust) sgd(fast) rmsprop(variable) adadelta(for sparse data)
    # activations: linear(lin,-inf/+inf) gelu(nonlin,-0.17/+inf) selu(nonlin,-a/+inf) sigmoid(nonlin,0/1) tanh(nonlin,-1/+1) softplus(nonlin,0/+inf) softsign(nonlin,-1/+1). linear=regression, relu|selu|gelu=general purpose, sigmoid|tanh=binary classification, softplus|softsign=multiclass classifiction
    # losses: hinge kld cosine_similarity mse msle huber binary_crossentropy sparse_categorical_crossentropy
    # metrics: mape mae accuracy top_k_categorical_accuracy categorical_accuracy

    from lib.supervised_learning import neural_network

    if '-s.nn=f' in o:  # feedforward
        model = neural_network.feedforward(feat, activ, nl, nu, nclass, outactiv)

    if '-s.nn=i' in o:  # imbalancenet
        model = neural_network.imbalancenet(feat, activ, nl, nu, nclass, outactiv)

    if '-s.nn=l' in o:  # lstm
        model = neural_network.lstm(feat, activ, nl, nu, nclass, outactiv, maxval)

    if '-s.nn=b' in o:  # bilstm
        model = neural_network.blstm(feat, activ, nl, nu, nclass, outactiv, maxval)

    if '-s.nn=g' in o:  # gru
        model = neural_network.gru(feat, activ, nl, nu, nclass, outactiv, maxval)

    if '-s.nn=r' in o:  # rnn
        model = neural_network.rnn(feat, activ, nl, nu, nclass, outactiv, maxval)

    if '-s.nn=c' in o:  # cnn
        x_train, x_test, model = neural_network.cnn(dshape, feat, activ, nl, nu, nclass, outactiv, maxval, x_train, x_test, xtrain_inst, xtest_inst)

    model.compile(optimizer=opt, loss=los, metrics=[metric])

    print('creating models on training set. max 100 epochs, stop after 5 epochs with no improvement')
    earlystop = TF.keras.callbacks.EarlyStopping(patience=5, monitor=metric)
    model.fit(x_train, y_train, epochs=100, callbacks=[earlystop], verbose=1)  #

    print('evaluating model on test set')
    model.evaluate(x_test, y_test, verbose=2)
    if target == 'c':
        y_pred = model.predict(x_test)
        y_pred = y_pred.argmax(axis=-1)
        y_test = PD.Series(y_test)  # make class predictions from functional model and count classes from test for conf matrix
    # print(y_pred)
    if target == 'r':
        y_pred = model.predict(x_test)
        y_pred = y_pred.flatten()
        # print(y_pred)
        y_test = PD.Series(y_test)
        # ymin=y_train.min()
        # ymax=y_train.max()
        # print(f"ymin={ymin} ymax={ymax}") #get weights from model
        # y_test=PD.Series(y_test)
        # diff=y_pred.flatten()-y_test
        # modeldiff=(diff / y_test)*y_test.mean()
        # y_pred=NP.abs(modeldiff) #make numeric predictions from functional model
        y_pred = PD.DataFrame(y_pred).to_numpy().flatten()

    if '-d.save' in o and '-s.nn' in o:  # save deep learning models
        opt = re.sub(r'-d.save|-e\..+|-d.viz| ', '', o)
        if target == 'c':
            opt = '-d.t=c' + opt
        if target == 'r':
            opt = f"-d.t=r" + opt
        model.save(f"{filename}{opt}.h5")
        print(f"model saved as {filename}{opt}.h5")

# ---time series forecasting
if '-t.' in o:
    dy_ = y_test[tscol]
    dy_ = PD.Series(dy_)  # take dates of test set
    x_train = x_train.set_index(tscol)
    y_train = y_train.set_index(tscol)  # set date as the instance index
    x_test = x_test.set_index(tscol)
    y_test = y_test.set_index(tscol)  # set date as the instance index
    # print(dir(SM.tsa))

if '-t.arma' in o:
    from lib.time_series_forecasting import auto_reg_moving_avg
    y_pred, model = auto_reg_moving_avg.apply_arma(y_train, y_test)

if '-t.arima' in o:
    from lib.time_series_forecasting import auto_reg_integrated_moving_avg
    y_pred, model = auto_reg_integrated_moving_avg.apply_arima(y_train, y_test)

if '-t.sarima' in o:
    from lib.time_series_forecasting import seasonal_auto_reg_integrated_moving_avg
    y_pred, model = seasonal_auto_reg_integrated_moving_avg.apply_sarima(y_train, y_test)

if '-t.hwes' in o:
    from lib.time_series_forecasting import holtwinters_exp_smoothing
    y_pred, model = holtwinters_exp_smoothing.apply_hwes(y_train, y_test)

if 'model' in locals() and '-t.' in o:
    if target == 'c':
        classes = model.classes_
    if '-d.save' in o and not '-t.nn' in o:  # save machine learning models
        opt = re.sub(r'-d.save|-e\..+|-d.viz| ', '', o)
        if target == 'c':
            opt = '-d.t=c' + opt
        else:
            opt = '-d.t=r' + opt
        joblib.dump(model, f"{filename}{opt}.h4")
        print(f"model saved as {filename}{opt}.h4")
        # af= open(f"{filename}-format4model.csv", 'w')
        # af.write(x_test.to_csv())
        # af.close()

if not 'y_pred' in locals():
    print('no supervised model trained. algorithm not found.')
    sys.exit()

# ---model details
if '-d.md' in o and not '-s.base' in o:
    print('model details:')
    # print(dir(model))
    if 'weights' in dir(model) and not '.knn' in o:  # nn
        mw = f" per.layer.node.weights:\n{model.weights}"
    if 'coef_' in dir(model) and not 'support_vectors_' in dir(model) and not '-s.xgb' in o:  # lr, lcm, sgd, nb
        mw = f" features * per.class.feat.weights {model.coef_} + per.class.intercept {model.intercept_}"
    if 'coefs_' in dir(model):  # mlp
        mw = f" per.feat.node.weights {model.coefs_} + per.node.intercept {model.intercepts_}"
    if 'support_vectors_' in dir(model):  # svm #svmp
        mw = f" points of support vectors:\n{model.support_vectors_}"
    if 'feature_importances_' in dir(model) and not '-s.dt' in o:  # rf ada, xgb
        mw = f" feature.importance:\n{model.feature_importances_}"
    if '.dt' in o:
        mwtree = tree.export_text(model)
        mw = f" rule.tree:\n{mwtree}"  # dt
    if '.knn' in o:
        mw = model.kneighbors(x_)
        mw = f" per.instance.neighbors.indexes:\n{mw[1]}\n per.neighbor.distances:\n{mw[0]}"  # knn

    if 'get_params' in dir(model):
        print(model.get_params())
    if '.nn' in o:
        print(model.summary())
    if 'mw' in locals():
        print(mw)

# ---evaluation

# if '-cv=' in o: #eval with cross validation (metrics: 'accuracy' 'balanced_accuracy')
#     if target=='c':
#          scores = SK.model_selection.cross_val_score(model, x_, y_, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
#          print(f'eval with {folds}-fold cross-validation BAL ACC= %.3f (+ - %.2f)' % (NP.mean(scores), NP.std(scores)))
#     if target=='r':
#          scores = SK.model_selection.cross_val_score(model, x_, y_, scoring='r2', cv=cv, n_jobs=-1)
#          print('eval with {folds}-fold cross-validation R2= %.3f (+ - %.2f)' % (NP.mean(scores), NP.std(scores)))


if '-s.' in o or '-t.' in o:  # if the task is supervised run evaluation
    x_test = x_test2  # restore x_test in its dataframe form
    if target == 'c':
        # scores=model.fit(x_train, y_train)
        # y_pred = model.predict_classes(x_test)
        # print(scores)
        # print(y_test)
        # print(y_pred)
        acc = metrics.balanced_accuracy_score(y_test, y_pred)
        print(f"eval predictions on test set. BAL ACC= {acc:.3f}")
        rr = metrics.classification_report(y_test, y_pred)
        print(rr)
        cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
        cm = PD.DataFrame(cm)
        print("confusion matrix:\n", cm)
        af = open('results.txt', 'a')
        af.write(f"\n\n{f}, {o} -->BAL ACC= {acc:.3f}\n{rr}\nconfusion matrix:\n{cm}")
        af.close()
    if target == 'r':
        # scores=model.fit(x_train, y_train)
        # y_pred=model.predict(x_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        y_scaled = preprocessing.MinMaxScaler().fit_transform(y_test.to_numpy().reshape(-1, 1))
        ys_ = PD.DataFrame(y_scaled)  # normalize ground truth
        p_scaled = preprocessing.MinMaxScaler().fit_transform(y_pred.reshape(-1, 1))
        ps_ = PD.DataFrame(p_scaled)  # normalize predictions
        r2 = metrics.r2_score(y_test, y_pred)
        # nmae=mean_absolute_percentage_error(y_test, y_pred)
        print(f'eval on test set. R2= {r2:.3f}, MAE= {mae:.3f}')  # eval with train-test split. balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, mean_absolute_percentage_error
        NP.set_printoptions(precision=2)
        if '-d.data' in o:
            print('predictions:')
            print(y_pred.flatten())
            print('ground truth:')
            print(y_test.to_numpy().flatten())
        af = open('results.txt', 'a')
        af.write(f"\n\n{f}, {o} --> R2= {r2:.3f}, MAE= {mae:.3f}\n")
        af.close()

# ---visualizations
# print("\n")
if '-d.viz' in o:
    if '-s.' in o:  # supervised scatterplots
        if task == 's' and target == 'c':
            pca = decomposition.PCA(2)
            projected = pca.fit_transform(x_test)
            MP.scatter(projected[:, 0], projected[:, 1], c=y_test, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
            MP.xlabel('component 1')
            MP.ylabel('component 2')
            MP.colorbar()
            MP.title('2D PCA ground truth test set')
            MP.savefig(fname='pca-test-space.png')
            MP.show()
            MP.clf()
            pca = decomposition.PCA(2)
            projected = pca.fit_transform(x_test)
            MP.scatter(projected[:, 0], projected[:, 1], c=y_pred, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
            MP.xlabel('component 1')
            MP.ylabel('component 2')
            MP.colorbar()
            MP.title('2D PCA predictions on test set')
            MP.savefig(fname='pca-testpred-space.png')
            MP.show()
            MP.clf()
        if task == 's' and target == 'r':
            MP.scatter(ys_, ps_, alpha=0.8)
            MP.xlabel('ground truth (norm)')
            MP.ylabel('predictions (norm)')
            # MP.show()
            # MP.clf()
            MP.scatter(ys_, ys_, alpha=0.2)
            # MP.legend(handles=['ys_', 'ps_'], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
            MP.title('predictions-truth scatterplot')
            MP.savefig(fname='target-pred-fit-space.png')
            MP.show()
            MP.clf()

    if '-t.' in o:  # multiple timeseries line plots
        # y_pred=PD.DataFrame(y_pred)
        # print(dy_)
        # print(y_test)
        # print(y_pred)
        # print(len(dy_))
        # print(len(y_test))
        # print(len(y_pred))
        MP.plot(dy_, y_test, marker='', color='orange', linewidth=2, label='ground truth')
        MP.plot(dy_, y_pred, marker='', color='blue', linestyle='dashed', linewidth=2, label='predictions')
        MP.title('test set linechart')
        MP.legend()
        MP.xticks(rotation=45, ha="right")
        MP.show()
        MP.clf()

    if '-x.' in o and 't_' in locals() and target == 'c':  # wordclouds
        print('terms most associated to each class')
        os.system("pip install wordcloud stop-words")
        from wordcloud import WordCloud
        from collections import defaultdict

        # from stop_words import get_stop_words
        words_weights = defaultdict(list)
        for idx, class_ in enumerate(y_pred):
            words_weights[class_].append(orig_t_[idx])
        # stopwords = get_stop_words('it')
        # stopwords.extend(["http", "https"])
        for k, v in words_weights.items():
            MP.title('Class \'%s\'' % k)
            wc = WordCloud(background_color='white', max_words=1000)
            wc.generate(" ".join(v))
            MP.imshow(wc, interpolation='bilinear')
            MP.axis('off')
            MP.savefig('wordcloud_%s.png' % k)
            MP.show()
            MP.clf()

timestamp = DT.datetime.now()
print(timestamp)
