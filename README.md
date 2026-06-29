# Effective Query-Independent Vector Pruning for Dense Retrieval
Gabriel Tolosa, Tomás Delvechio and Esteban Rissola

**Abstract**
> Nowadays, one approach to effective document retrieval is the so-called dense retrieval, based on encoding models that transform text documents and queries into embeddings. These models generate vectors of varying sizes, which offer different effectiveness but also impact on the retrieval efficiency. One approach to deal with this problem is to reduce the number of dimensions considered for the retrieval task. Recent studies propose methods to reduce embedding size without performance degradation in query-dependent setups, which require additional computations at query time. In this work, we address the problem of ranking dimensions of dense vectors by proposing and evaluating methods coming from different conceptual backgrounds. The methods estimate the contribution of each dimension in a query-independent manner, allowing for vector pruning and thereby improving retrieval time. The evaluation considers four models to generate embeddings (with size variants) and two million-document collections. Our experiments reveal that some of the proposed methods can reduce the size of vectors by up to 90% in specific dense retrieval models, while maintaining a competitive retrieval quality. We also show that the first dimensions contribute the most to total effectiveness performance when ranking the dimensions with the best-performing methods. 


### Source code repository: 

```
/bin             # Scripts use for experiments 
/config          # Example configuration files
/runs            # Output directory
/vectors         # Dense vectors directory
```
 
Examples

**Computing embeddings**
```
# python3 ./bin/01-compute_embeddings.py -c <collection name> -m <embedding model> -f <corpus file> -o <output directory>
python3 ./bin/01-compute_embeddings.py -c msmarco -m msmarco-distilbert-base-tas-b -f ./collections/collection.tsv -o ./vectors/
```

**Computing dimension ranking**
```
# python3 ./bin/02-compute_feats_importance_stat.py -v <vectors file> -m <method> -o <output file>
python3 ./bin/02-compute_feats_importance_stat.py -v ./vectors/msmarco-distilbert-base-tas-b_embeddings.full.npy -m var -o ./vectors/dimension_ranking/msmarco-distilbert-base-tas-b_embeddings.var.npy
```

**Pruned retrieval**
```
# python3 ./bin/03-pruned_retrieval.py <configuration file>
python3 ./bin/03-pruned_retrieval.py ./config/msmarco_dev_small_bge-large.ini
```
The parameters of the runs are contained inside configuration files (which include the corresponding paths for vectors files and outputs)




