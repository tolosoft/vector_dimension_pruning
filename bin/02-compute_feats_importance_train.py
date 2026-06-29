import sys
import pandas as pd
import numpy as np
import argparse
import random
from time import process_time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
#
#
def get_random_notrelevant(relevant):
    nr = {}
    mmin = min(relevant.keys())
    mmax = max(relevant.keys())
    #
    while len(nr) < len(relevant):
        nn = random.randint(mmin, mmax)
        if nn not in relevant:
            nr[nn] = 1
    return nr
#
#
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vectors",    help = "Vectors file", required=True)
    parser.add_argument("-j", "--judgements", help = "Qrels file", required=True)
    parser.add_argument("-o", "--output",     help = "Output dir", default="")
    parser.add_argument("-m", "--method",     help = "Method for computing feature importances", required=True, choices=["rf", "perm"])
    parser.add_argument("-s", "--qrels_sep",  help = "Qrels file separator", required=True, choices=["blank", "tab"])
    args = parser.parse_args()
    #
    vectors = np.load(args.vectors)
 
    docs_file = args.vectors.replace("embeddings.full.npy", "docids.npy")
    doc_ids = np.load(docs_file, allow_pickle=True)

    sep = " "
    if args.qrels_sep == "tab":
        sep = '\t'
    qrels_file = args.judgements
    judgments = pd.read_csv(qrels_file, delimiter=sep, header=None).rename(columns={0:'qid', 1:'iteration', 2: 'docid', 3: 'relevancy grade'})[['qid', 'docid', 'relevancy grade']]  
    
    # Adapt for dbpedia
    # judgments = pd.read_csv(qrels_file, delimiter=sep).rename(columns={'query-id':'qid', 'corpus-id':'docid', 'score': 'relevancy grade'})
        
    #
    not_relevant = dict.fromkeys(list(judgments[(judgments['relevancy grade']==0)]['docid']), 0)
    relevant = dict.fromkeys(list(judgments[(judgments['relevancy grade']>0)]['docid']), 1)
    #
    if (len(not_relevant) == 0):
        not_relevant = get_random_notrelevant(relevant)
    #
       
    X = []
    y = []
    for i, did in enumerate(doc_ids):
        did = int(did)
        if (did in relevant) and (did not in not_relevant):
            X.append(vectors[i])
            y.append(1)            
        #
        if (did in not_relevant) and (did not in relevant):
            X.append(vectors[i])
            y.append(0) 
    print (len(X))

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X.columns = ["f"+str(i) for i in range(0, X.shape[1])] 
    #
    print (X.shape, y.shape)
    columns_order = []
    if args.method == "rf":
        forest = RandomForestClassifier(random_state = 0)
        forest.fit(X, y)
        #
        importances = forest.feature_importances_
        columns_order = list(range(1, len(importances)+1))
        imp_tuples    = zip(importances, columns_order)
        #
        columns_order = []
        for t in sorted(imp_tuples, reverse=True):
            columns_order.append((t[1]-1))
        #
    #
    if args.method == "perm": 
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 42)
        forest = RandomForestClassifier(random_state = 0)
        forest.fit(X_train, y_train)
        #
        result_permut = permutation_importance(
            forest, X_test, y_test, n_repeats = 10, random_state = 42, n_jobs=2
        )
        importances   = result_permut.importances_mean
        columns_order = list(range(1, len(importances)+1))
        imp_tuples = zip(importances, columns_order)
        #
        columns_order = []
        for t in sorted(imp_tuples, reverse=True):
            columns_order.append((t[1]-1))
        #
    #

    if len(columns_order) == len(vectors[0]):
        print (importances)
        if args.output != "":
            np.save(args.output, columns_order)    
    else:
        print ("Houston! Len doesn´t match!")
    
    sys.exit()