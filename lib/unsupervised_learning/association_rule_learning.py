def arl(x_):
    import numpy as NP
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    import pandas as PD
    import datetime as DT
    import sys

    xn_ = x_.to_numpy()
    xn_ = xn_.flatten()
    xn_ = NP.array(xn_, dtype=NP.str)
    print('apply association rule learning (market basket analysis).\ntheory: https://en.wikipedia.org/wiki/Association_rule_learning \ndocs: https://github.com/rasbt/mlxtend')
    xn_ = NP.char.split(xn_, sep=' ')

    te = TransactionEncoder()
    te_ary = te.fit(xn_).transform(xn_)
    df = PD.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    # print(frequent_itemsets)
    # frequent_itemsets = fpmax(df, min_support=0.01, use_colnames=True)
    results = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    af = open('analysis.txt', 'a')
    af.write(results.to_string() + "\n\n")
    af.close()
    print(results)
    print('results printed in analysis.txt file')
    timestamp = DT.datetime.now()
    print(f"-u.arl stops other tasks\ntime:{timestamp}")
    sys.exit()