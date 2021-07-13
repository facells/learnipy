# LEARNIPY
making machine learning easy for everyone

written with ♥ by Fabio Celli, 

email: fabio.celli.phd@gmail.com

twitter: @facells

tested in: Google colab

License: MIT (Commercial use,  Modification, Distribution, Private use are permitted, Liability is yours, No software warranty)



### 2) USAGE 

 1. to train a model: load learnipy and traindata (.csv or .zip) in https://colab.research.google.com/ , testdata is optional, and type

 >%run learnipy.py 'options' traindata [testdata]

  where 'options' is a string containing the operations, defined at paragraph 4.

 2. to make predictions on new data: load model (.h5) and data (.csv or .zip) in colab and type

 >%run learnipy.py '-d.pred' model.h5 [testdata]



### 3) DATA FORMATTING

 data.csv must be a , separated file, 

 the target column must be named 'class'

 the text column must be named 'text'

 

 data.zip must contain .png .jpg files.

 the files names must be , separated. example: imgID,class,.jpg



### 4) OPTIONS

#### data management

 * __-d.c=l__      manually define type of target class. params: l=label, 0=number

 * __-d.gen=132__  generate dataset, create gen.csv.params 1=num instances x1000, 3=num features x10, 2=num informative features x10

 * __-d.viz__      visualize class distribution and pca-projected 2d data scatterplot

 * __-d.fdst__     print info on feature distribution

 * __-d.data__     show preview of processed data

 * __-d.cnorm__    normalize numeric class

 * __-d.save__     save model as .h4 (machine learning) or .h5 (deep learning) file

 * __-d.pred__     use model to make predictions on new data

#### preprocessing

 * __-p.rand__     randomize instances in the training set

 * __-p.norm__     feature normalization, range 0,1 (applied by default with sgd and nb)

 * __-p.tl__       text to lowercase

 * __-p.tc__       clean text from non alphanum char and multiple spaces

#### feature reduction

 * __-r.svd=5__    singular value decomposition. turn sparse label binarized matrix to dense and sync. param: 5=number of features

 * __-r.lsa=5__    latent semantic analysis. turn sparse word/char matrix to dense and sync. param: 5=number of features

#### text feature extraction

 * __-x.ng=23cf__  ngrams. turn text to word|char ngrams freq|tfidf matrix and apply lda. params: 2=min, 3=max, c=chars|w=words, f=freq|t=tfidf

 * __-x.tm=5__     token matrix. turn text into word frequency matrix. 5=integer, number of features

 * __-x.ts=5__     token sequences. turn text into sequence word index feature matrix. columns are padded sequences of words. 5=integer, number of features 

 * __-x.w2v__      extract word2vec dictionary and save it. also visualize a 2d word2vec space

 * __-x.d2v=5__    turn text into doc2vec dense feature matrix. 5=integer, number of features

 * __-x.bert__     turn text into multi-language bert 768-length vectors

 * __-x.d=d.lex__  turn text into vectors from custom dictionary dimensions. d.lex is an external lexical resource, comma separated

#### unsupervised learning

 * __-u.km=2__     processed feature analysis with kmeans centroid clustering. add a new colum to dataset. results in analysis.txt. params: 2=num clusters

 * __-u.optics__   processed feature analysis with optics density clustering. add a new colum to dataset. results in analysis.txt. 

 * __-u.msh__      processed feature analysis with mshift density clustering. add a new colum to dataset. results in analysis.txt.

 * __-u.arm__      association rule mining. default algorithm: apriori. prints results in analysis.txt

 * __-u.corr__     processed feature analysis with pearson correlations. prints results in analysis.txt

#### supervised learning

 * __-s.base__     majority baseline for classification and regression

 * __-s.nb__       probabilistic models. complement naive bayes for classification, bayes ridge for regression

 * __-s.lr__       linear regression and logistic regression

 * __-s.lcm __     linear combination models, linear discriminant classifiction and partial least squares regression

 * __-s.sgd__      linear modeling with stochastic gradient descent

 * __-s.knn__      k nearest neighbors classification and regression

 * __-s.dt__       decision trees and regression trees

 * -s.mlp__      multi layer perceptron

 * __-s.svmp=3__   svm with poly kernel. param p=polynomial kernel. 3=dimensions of polynomial kernel, 

 * __-s.svm__      svm with radial basis function.

 * __-s.rf__       ensemble learning, random forest

 * __-s.ada__      ensemble learning, adaboost based on samme.r algorithm

 * __-s.xgb__      ensemble learning, xgboost

 * __-s.nn=f__     deep learning. params: f=feedfwd|i=imbalance|r=rnn|l=lstm|b=bilstm|g=gru|c=cnn. 

#### evaluation

 * __-e.tts=0.2__  train-test split. params 0.2=20% test split. param not valid if test set is provided.
