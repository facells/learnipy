# LEARNIPY
* version 0.6
* making machine learning easier
* written with ♥ by Fabio Celli, 
* email: fabio.celli.phd@gmail.com
* twitter: @facells
* tested in Google colab
* License: MIT (Commercial use,  Modification, Distribution, Private use are permitted, Liability is yours, No software warranty)
* Conditions: Report the following license and copyright notice with code.
"Copyright (c) 2021 Fabio Celli.
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
### 1) AIM 
* we want to make machine learning accesible and easy to use for everyone. 
* We want a system that is self-contained (one file), portable, 100% written in Python.
### 2) DATA FORMATTING
* data.csv must be a comma-separated file (,)
* the target column can be named 'class' in the .csv file or defined with -d.c= option
* the text column can be named 'text' in the .csv file or defined with -d.s= option
* data.zip must contain .png or .jpg files. the files names must be comma-separated. example: imgID,class,.jpg
### 4) USAGE 
* to train a model: %run learnipy.py 'options' traindata [testdata], for example
 >%run learnipy.py '-d.t=c -x.tm=700 -d.viz -s.nn=f' traindata.csv
* 'options' is a string containing the operations, defined at paragraph 4.
* yourdata.csv can be a .csv for tabular and text data or .zip for pictures.
* [testdata] is optional, if given is used as a test set, if not the training set is split
* to make predictions on new data: %run learnipy.py '-d.pred' model testdata, for example
 >%run learnipy.py '-d.pred' model.h5 testdata.csv
* models can have .h5 (deep learning) or .h4 (machine learning) extension
* try it on https://colab.research.google.com/drive/1DfDp2VFaTTMz_B6uLrOdWKQkrer32S9M?usp=sharing
### 4) DOCUMENTATION
#### data management
* -d.t=c|r    *define type of task. c=classification, r=regression*
* -d.x=n,m,o  *define the columns to exclude. n,m,o=names of columns to exclude*
* -d.s=n      *define the string column treated as text. n=name of text column
* -d.c=n      *define the column of the target class. n=name (for .csv) or index (for .zip) of class column*
* -d.r=0      *do not use feature reduction, keep original features (not applicable with -d.save)*
* -d.z=30     *define custom resize of pictures. 30=size 30x30*
* -d.m=1      *fill class missing values. 1=replace all missing values in class with mean/mode (otherwise are deleted by default)*
* -d.viz      *print pca-projected 2d data scatterplot and other visualizations*
* -d.md       *model details. prints info on algorithm parameters and data modeling*
* -d.fdst     *print info on feature distribution*
* -d.data     *show preview of processed data*
* -d.save     *save model as .h4 (machine learning) or .h5 (deep learning) file*
* -d.pred     *use model to make predictions on new data*
* -d.export=f *export processed data in csv. f=filename.csv*
#### data generation
* -g.d=132    *generate dataset, create gen.csv. 1=num instances x1000, 3=num features x10, 2=num informative features x10*
#### preprocessing
* -p.ir       *instance position randomization, applies to the training set*
* -p.cn       *class normalize. turn numeric class to range 0-1*
* -p.fn       *feature normalize, turn features to range 0-1 (applied by default with some nn, sgd and nb)*
* -p.tl       *text to lowercase*
* -p.tc       *text cleaning. removes non alphanum char and multiple spaces*
* -p.trs      *text regex stopwords. removes words from length 1 to length 3*
* -p.tsw=a,b  *text stopwords. removes stopwords, a,b=stopwords list, no spaces allowed.*
#### feature reduction
* -r.svd=5    *singular value decomposition. turn sparse label matrix to dense and sync. 5=number of features*
* -r.lsa=5    *latent semantic analysis. turn sparse word/char matrix to dense and sync. 5=number of features*
#### feature extraction
* -x.ng=23cf  *ngrams. turn text to word|char ngrams freq|tfidf matrix and apply lsa. 2=min, 3=max, c=chars|w=words, f=freq|t=tfidf*
* -x.tm=5     *token matrix. turn text into word frequency matrix. 5=number of features*
* -x.ts=5     *token sequences. columns are padded sequences of words. 5=number of features* 
* -x.cm=5     *char matrix. turn text into character frequency matrix. 5=number of features*
* -x.d2v=5    *(deprecated. will be removed) turn text into doc2vec word-context dense feature matrix. 5=number of features*
* -x.bert     *extract 768 features from text to a dense matrix with multi-language bert transformer model*
* -x.mobert   *extract 512 features from text to a dense matrix with multi-language mobile bert transformer model*
* -x.d=e      *extract features from custom dictionary. e=dictionary. check https://github.com/facells/learnipy/tree/main/resources
#### unsupervised learning
* -u.km=2     *kmeans, centroid clustering. add a new colum to dataset. results in analysis.txt. 2=num clusters*
* -u.optics   *optics, density clustering. add a new colum to dataset. results in analysis.txt*
* -u.msh      *mshift, density clustering. add a new colum to dataset. results in analysis.txt*
* -u.ap       *affinity propagation exemplar clustering. add a new colum to dataset. results in analysis.txt*
* -u.som      *self organising map, neural network clustering. add a new colum to dataset. results in analysis.txt*
* -u.w2v[=15] *word2vec dictionary from text, pca-2d word2vec space. 1=words to filter x10, 5=words to visualize x10*
* -u.arl      *association rule learning with apriori. prints results in analysis.txt*
* -u.corr     *feature analysis with pearson correlations. prints results in analysis.txt*
#### outlier detection
* -o.if       *isolation forest. find and remove outliers using random forest regions*
* -o.mcd      *minimum covariance determinant with ellipsis envelope. find and remove outliers using gaussian distribution*
* -o.lof      *local outlier factor. find and remove outliers using optics less dense regions*
#### supervised learning
* -s.base     *majority baseline for classification and regression*
* -s.nb       *probabilistic models. complement naive bayes for classification, bayes ridge for regression*
* -s.lr       *linear regression and logistic regression*
* -s.lcm      *linear combination models, linear discriminant classifiction and partial least squares regression*
* -s.sgd      *linear modeling with stochastic gradient descent*
* -s.knn      *k nearest neighbors classification and regression*
* -s.dt       *decision trees and regression trees*
* -s.mlp      *multi layer perceptron*
* -s.svm[=p3] *svm (rbf kernel by default). p=polynomial kernel|r=rbf kernel (default), 3=kernel degrees*
* -s.rf       *ensemble learning, random forest*
* -s.ada      *ensemble learning, adaboost based on samme.r algorithm*
* -s.xgb      *ensemble learning, xgboost*
* -s.nn=f[51] *deep learning. f=feedfwd|i=imbalance|r=rnn|l=lstm|b=bilstm|g=gru|c=cnn. 5= x10 units, 1=num layers*
#### time series forecasting
* -t.arma     *auto regression moving average*
* -t.arima    *auto regression integrated moving average*
* -t.sarima   *seasonal auto regression integrated moving average*
* -t.hwes     *Holt-Winters exponential smoothing*
#### evaluation
* -e.tts=0.2  *train-test split. 0.2=20% test split. ignored if test set is provided*
### 5) CHANGELOG
* v0.0: developed the main features
* v0.1: added -u.corr, -u.arl, -x.w2v, -x.d2v, -s.sgd, -s.xgb, .zip input, -s.nn=c
* v0.2: added -x.bert, -x.tm, -x.ts, improved -s.nn, removed -e.cv (cross validation), fixed bug on text reading
* v0.3: improved -x.bert, -x.d and -d.viz, added -d.c, -d.s, -d.m, -d.r, -d.x, changed -d.gen to -g.d
* v0.4: added -d.export -g.mct, -u.som, -d.md, included -s.psvm in -s.svm, added wiki links, moved -u.w2v
* v0.5: added -p.trs, -p.tsw, -o.if, -o.mcd, -o.lof, -u.ap, fixed bug on .zip reading, improved -u.corr
* v0.6: improved anomaly detection evaluation, added -t., -x.mobert
### 6) TO DO LIST
* links to sklearn and tensorflow documentation for algorithms
* -g.mct (markov chains generated text)
* -g.gpt (gpt generated from text)
* improve -u.corr  (correlation ranking)
* remove -x.d2v because it is not replicable
* -x.gap global average pooling image feature extraction
* table to image extraction
* improve test set input
* -u.gxm expectation maximisation
