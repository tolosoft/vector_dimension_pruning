import os
import sys
import pandas as pd
import configparser
import logging
from time import process_time
import faiss
#
#
def read_config(config):
    config_file = ""
    try:
        config_file = sys.argv[1]
    except IndexError:
        print ("Run: ", sys.argv[0],"<config_file>")
        sys.exit()
    config.read(config_file)
    config["Logging"]["config_file"] = config_file
    return config
#
#
def load_ids_mappings(doc_vectors_file, qry_vectors_file):
    docs_file = doc_vectors_file.replace("embeddings.full.npy", "docids.npy")
    doc_ids = np.load(docs_file)
    print ("Loading: ", docs_file)   
    #    
    qrys_file = qry_vectors_file.replace("embeddings.full.npy", "docids.npy")
    qry_ids = np.load(qrys_file)   
    print ("Loading: ", qrys_file)
    #
    return doc_ids, qry_ids
#
#
def do_id_mapping(item, do_map, doc_ids, qry_ids, this_id):
    #
    if ((item == "query") and (do_map=="queries" or do_map=="both")):
        this_id = qry_ids[this_id]

    if ((item == "doc") and (do_map=="docs" or do_map=="both")):
        this_id = doc_ids[this_id]
    #
    return this_id
#
#
def faiss_index(vectors, index_type):
    # Start time
    time_start = process_time()
    #
    if index_type[0:4] == "flat":
        d = vectors.shape[1]
        #
        if index_type == "flatIP":
            indexFlat = faiss.IndexFlatIP(d)
        if index_type == "flatL2":
            indexFlat = faiss.IndexFlatL2(d)
        #    
        indexFlat.add(vectors)
    #
    execution_time = process_time() - time_start
    return indexFlat, execution_time 
# 
#
def faiss_search(thisIndex, qry_vectors_cropped, k):
    # Start time
    time_start = process_time()
    #
    sim_val, sim_ids = thisIndex.search(qry_vectors_cropped, k)
    #
    execution_time = process_time() - time_start
    return sim_val, sim_ids, execution_time 
#
#
def calc_mrr(search_results, qrels, nqueries):
    labeled_search_results = search_results.merge(qrels, how='left', on = ['qid', 'docid']).fillna(0)
    relevances_rank = labeled_search_results.groupby(['qid', 'relevancy grade'])['rank'].min()
    ranks = relevances_rank.loc[:, 1]
    reciprocal_ranks = 1 / (ranks)
    return reciprocal_ranks.sum() / nqueries
#
#-------------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read configfile 
    config = configparser.ConfigParser()
    global_config = read_config(config)
    #
    doc_vectors_file = global_config["Data"]["doc_vectors"]
    qry_vectors_file = global_config["Data"]["qry_vectors"]
    index_type       = global_config["Index"]["type"]
    feats_file       = global_config["Feature_Selection"]["features_prefix"]
    feats_labels     = global_config["Feature_Selection"]["features_labels"].split(",")
    feats_percent    = global_config["Feature_Selection"]["features_percentages"].split(",") if feats_labels != [] else [100]
    save_iruns       = global_config["Feature_Selection"]["save_individual_runs"]
    k                = int(global_config["Retrieval"]["Top_k"])
    nqueries         = int(global_config["Retrieval"]["nqueries"])
    norm_queries     = global_config["Retrieval"]["norm_queries"]
    output_prefix    = global_config["Retrieval"]["output_file_prefix"]
    qrels_file       = global_config["Evaluation"]["qrels_file"]
    qrels_file_sep   = global_config["Evaluation"]["qrels_file_separator"]
    map_ids          = global_config["Evaluation"]["map_ids"]
    log_file         = global_config["Logging"]["log_dir"]
    #-----------------------------------------------------------------------------------

    log_file = log_file + os.path.basename(sys.argv[0]) + ".log"
    log_pid = "pid"+str(os.getpid())

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        filemode="a",
        format="{asctime} * {levelname} * {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG,
    )
    #-----------------------------------------------------------------------------------
    output_prefix = output_prefix + "." + index_type + ".nq-" + norm_queries + ".k" + str(k) 
    #
    logging.info(log_pid+"*"+config["Logging"]["config_file"])
    print (log_pid)
    print (output_prefix)
    print (feats_percent)

    #
    # 
    doc_vectors = np.load(doc_vectors_file)
    print (doc_vectors_file, doc_vectors.shape)

    # 
    qry_vectors = np.load(qry_vectors_file)
    print (qry_vectors_file, qry_vectors.shape)
    
    if norm_queries == "True":
        faiss.normalize_L2(qry_vectors)
        print ("L2 normalize queries")

    doc_ids = []
    qry_ids = []
    if map_ids != "none":
        # Mappings docids and qryids
        doc_ids, qry_ids = load_ids_mappings(doc_vectors_file, qry_vectors_file)
    #
    compute_mrr = False
    if qrels_file != "":
        sep = " "
        if qrels_file_sep == "tab":    # MS MARCO
            sep = '\t'
        #
        judgments = pd.read_csv(qrels_file, delimiter=sep, header=None).rename(columns={0:'qid', 1:'iteration', 2: 'docid', 3: 'relevancy grade'})[['qid', 'docid', 'relevancy grade']]
        compute_mrr = True
        #   
    #
    if nqueries == 0:
        nqueries = len(qry_vectors)
    else:
        qry_vectors = qry_vectors[0:nqueries]
    #
    #--------------------------------------------------------------------------------------------------------------------------------
    
    for this_feat in feats_labels:
        # Dataframe pfor results summary
        df_results = pd.DataFrame(columns = ['per_features', 'indexing_time', 'retrieval_time', 'mrr@10', 'mrr@100'])
        #
        print ("Features:", this_feat)
        logging.info(log_pid+"*Using feats: " + this_feat)
        
        if this_feat != "":
            feats_filename = feats_file + "." + this_feat + ".npy"     
            if os.path.exists(feats_filename):
                feats_info = np.load(feats_filename)
                feats_percent_list = feats_percent
            else:
                logging.warning(log_pid+"*Houston! "+feats_filename+" not found!")
                this_feat = ""
                feats_percent_list = [100]
        #   
        for percentage in feats_percent_list:
            logging.info(log_pid+"*"+"Running with "+ str(percentage) + "%")
            print ("Running with", percentage, "%")
            #
            if this_feat != "":
                # Calculo columas a usar
                nro_columns = int(len(feats_info) * int(percentage) / 100)              
                sel_columns = feats_info[0:nro_columns]
                # Crop of document vectors
                doc_vectors_cropped = np.ascontiguousarray(doc_vectors[:,sel_columns])
                # Crop of query vector
                qry_vectors_cropped = np.ascontiguousarray(qry_vectors[:,sel_columns])
            else:
                doc_vectors_cropped = doc_vectors
                qry_vectors_cropped = qry_vectors
            #
            # Indexing 
            thisIndex, indexingTime = faiss_index(doc_vectors_cropped, index_type)
            # Retrieval
            sim_val, sim_ids, searchTime = faiss_search(thisIndex, qry_vectors_cropped, k)
            # 
            # Store runs across all % of features used
            rs_data = [] 
            for qi, rs in enumerate(sim_ids):
                for j, dj in enumerate(rs, start=1):
                    did = do_id_mapping("doc", map_ids, doc_ids, qry_ids, dj)            
                    qid = do_id_mapping("query", map_ids, doc_ids, qry_ids, qi)
                    #    
                    rs_data.append([j, did, qid])   # rank     docid      qid
                    #
                #
            #
            search_results = pd.DataFrame(rs_data, columns=['rank', 'docid', 'qid'])  
            #
            if save_iruns == "True":
                output_file = output_prefix + "." + this_feat + "." + str(percentage) + ".csv"
                search_results.to_csv(output_file, index=False)
            #
            mrr10  = -1
            mrr100 = -1
            if (compute_mrr):
                search_results["qid"]   = search_results["qid"].astype(np.int64)
                #
                try:
                    this_run_k = search_results[search_results['rank'] <= 10]
                    mrr10 = calc_mrr(this_run_k, judgments, nqueries)
                    #
                    this_run_k = search_results[search_results['rank'] <= 100]
                    mrr100 = calc_mrr(this_run_k, judgments, nqueries)
                except:
                    pass
            #        
            df_results.loc[len(df_results.index)] = [percentage, indexingTime, searchTime, mrr10, mrr100]
            print ()
            #   
        print (df_results.head(20))
        output_file = output_prefix + "." + this_feat + "." + "summary.csv"
        df_results.to_csv(output_file, index=False)
    #
    logging.info(log_pid+"*"+"Normal exit")
    sys.exit()
