import sys
import pandas as pd
import numpy as np
import random
import argparse
from time import process_time
import networkx as nx
#
#
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vectors", help = "Vectors file", required=True)
    parser.add_argument("-o", "--output",  help = "Output dir", default="")
    parser.add_argument("-m", "--method",  help = "Method for computing feature importances", required=True, choices=["corsum", "mst", "var", "rnd"])
    args = parser.parse_args()
    #
    vectors = np.load(args.vectors)
    print (len(vectors), len(vectors[0]))
    #
    methods_need_correlation = ['corsum', 'mst']

    if args.method in methods_need_correlation:
        nro_columns = len(vectors[0])
        sel_columns = list(range(0, nro_columns))
        #
        v2 = vectors[:,sel_columns]
        nro_filas = len(vectors)
        #
        v3 = v2[0:nro_filas].T
        #
        print (len(v3), len(v3[0]))
        #
        # Calculo el coeficiente de correlación
        corr_matrix = np.corrcoef(v3)
        
        #print (corr_matrix)
        print ("Correlation matrix shape: ", corr_matrix.shape)
        
        if args.method == "corsum":
            #
            corr_sum = np.empty(nro_columns, dtype=float) 
            for i in range(0,nro_columns):
                tmp_sum = 0 
                for j in range(0,nro_columns):
                    if i != j:
                        #print (i,j,corr_matrix[i][j])
                        tmp_sum = tmp_sum + corr_matrix[i][j]
                #
                corr_sum[i] = tmp_sum
            #
            #print (corr_sum)   
            #
            column_order  = list(range(1, len(corr_sum)+1))
            corr_tuples = zip(corr_sum, column_order)
            corr_sorted = sorted(corr_tuples, reverse=True)
            #
            column_order = []
            for t in corr_sorted:
                column_order.append((t[1]-1))
            #
        #
        
        if args.method == "mst":
            cm_rows = corr_matrix.shape[0]   
            #
            G = nx.Graph()
            #
            for i in range(0, cm_rows):
                for j in range(i+1,cm_rows):
                    #print (i, j, cm[i][j])
                    G.add_edge(i, j, weight=corr_matrix[i][j])
            #
            T = nx.minimum_spanning_tree(G, algorithm='kruskal')
            #
            edges = sorted(T.edges(data=True))
            tuples = []
            #
            for e in edges:
                #print (e[0], e[1], e[2]["weight"])
                tuples.append((e[2]["weight"], (e[0], e[1])))
                    #    
            already_seen = {}
            column_order = []
            #
            for t in sorted(tuples):
                n1 = t[1][0]
                n2 = t[1][1]
                if n1 not in already_seen:
                    column_order.append(n1)
                    already_seen[n1] = 1
                if n2 not in already_seen:
                    column_order.append(n2)
                    already_seen[n2] = 1
            #        
        #            
    
    
    if args.method == "var":
        variance_columns = np.var(vectors, axis = 0)
        print (len(variance_columns))
        #
        number_colums   = list(range(1, len(variance_columns)+1))
        variance_tuples = zip(variance_columns, number_colums)
        #
        column_order = []
        for t in sorted(variance_tuples, reverse=True):
            column_order.append((t[1]-1))


    if args.method == "rnd":
        vectors_dim = len(vectors[0])
        #   
        column_order = list(range(0, vectors_dim))
        random.shuffle(column_order)
    #
    # 
    print (column_order)
    print (len(column_order))
    if args.output != "":
        np.save(args.output, column_order)
    #

    sys.exit()