# Effective Query-Independent Vector Pruning for Dense Retrieval
Gabriel Tolosa, Tomás Delvechio and Esteban Rissola

Source code repository: 

```
/bin             # Scripts use for experiments 
/config          # Example configuration files
/runs            # Output directory
/vectors         # Dense vectors directory
```
 
Examples

**Computing embeddings**
```
python3 ./bin/compute_embeddings.py -c msmarco -m msmarco-distilbert-base-tas-b -o ./vectors/
```

**Computing dimnesion ranking**
```
python3 ./bin/compute_feats_importance_stat.py -v ./vectors/msmarco-distilbert-base-tas-b_embeddings.full.npy -m var -o ./vectors/dimension_ranking/msmarco-distilbert-base-tas-b_embeddings.var.npy
```

**Pruned retrieval**
```
python3 ./bin/pruned_retrieval.py ./config/msmarco_dev_small_bge-large.ini
```





