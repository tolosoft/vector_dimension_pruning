import sys
import pandas as pd
import torch
import argparse
from time import process_time
#
import pyterrier as pt
from pyterrier.measures import *
from sentence_transformers import SentenceTransformer, util
#
#
def get_corpus_text(corpus_name, corpus_file):
    #
    doc_txt = []
    doc_ids = []
    #
    if corpus_name == "msmarco": 
        with open(corpus_file, "r") as fin:
            for q in fin:        
                id, qy = q.strip().split("\t")        
                doc_txt.append(qy)
                doc_ids.append(id)        
        #  
    #
    if corpus_name == "msmarco_dev_small":
        with open(corpus_file, "r") as fin:
            for q in fin:        
                id, qy = q.strip().split("\t")        
                doc_txt.append(qy)
                doc_ids.append(id)
        #        
    #
    if corpus_name == "dbpedia":
        df = pd.read_json(corpus_file, lines=True)
        doc_txt = list(df['title'] + " " + df['text'])
        doc_ids = list(df['_id'])#
    #
    if corpus_name == "dbpedia_queries":
        df = pd.read_json(corpus_file, lines=True)
        doc_txt = list(df['text'])
        doc_ids = list(df['_id'])#
    #
    return doc_txt, doc_ids, len(doc_ids)
#
#
# Modelos: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
models = {
    'msmarco-distilbert-base-tas-b': 'sentence-transformers/msmarco-distilbert-base-tas-b',
    'ance': 'sentence-transformers/msmarco-roberta-base-ance-firstp',               
    'bge-small': 'BAAI/bge-small-en-v1.5',
    'bge-base': 'BAAI/bge-base-en-v1.5',
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'modern-bert-base':'answerdotai/ModernBERT-base',
    'modern-bert-large':'answerdotai/ModernBERT-large'    
}   
#

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument("-c", "--corpus",  help = "Corpus name", required=True, choices = ["msmarco", "msmarco_dev_small", "dbpedia", "dbpedia_queries"])
    parser.add_argument("-m", "--model",   help = "Model name",  required=True, choices = models.keys())
    parser.add_argument("-o", "--output",  help = "Output dir",  required=True)
    parser.add_argument("-f", "--file",    help = "Corpus file", default="")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Documents (text, ids)
    doc_text, doc_ids, ndocs = get_corpus_text(args.corpus, args.file)
    #
    print ("Corpus:", args.corpus, ndocs, "docs. Model:", args.model, models[args.model])
    
    # Model
    model = SentenceTransformer(models[args.model])
    
    # Parameters
    batch_size = 10000
    doc_ini = 0
    doc_fin = batch_size
    iter = 0
    #
    all_doc_embeddings = np.empty((0, 1024), dtype = np.float32)

    # Record start time
    time_start = process_time()
    while True:
        print ("Doing", iter, doc_ini, doc_fin)
        batch_embeddings = model.encode(doc_text[doc_ini:doc_fin], convert_to_tensor=True)
        doc_embeddings = np.array(batch_embeddings.cpu())
        all_doc_embeddings = np.append(all_doc_embeddings, doc_embeddings, axis=0)
        #-------------------------------------------------------------
        if doc_fin >= ndocs:
            break
        #
        iter += 1
        doc_ini += batch_size
        doc_fin += batch_size
        if doc_fin > ndocs:
            doc_fin = ndocs
        #
    # Time duration
    time_duration = process_time() - time_start
    print(f'{time_duration:.3f} seconds')
    #
    # Save vectors
    embeddings_file = args.output+"/"+args.corpus+"-"+args.model+'_embeddings.full'
    np.save(embeddings_file, all_doc_embeddings)
    #
    # Save docids
    docids_file = args.output+"/"+args.corpus+"-"+args.model+'_docids.npy'
    np.save(docids_file, doc_ids)
#
#