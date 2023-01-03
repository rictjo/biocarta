# Biocarta
Creating Cartographic Representations of Biological Data
[![DOI](https://zenodo.org/badge/578172132.svg)](https://zenodo.org/badge/latestdoi/578172132)

# Installation
```
pip install biocarta
```

# Example code
```
if __name__ == '__main__' :
    from biocarta.quantification import full_mapping
    #
    adf = pd.read_csv('analytes.tsv',sep='\t',index_col=0)
    #
    # WE DO NOT WANT TO KEEP POTENTIALLY BAD ENTRIES 
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy()
    #
    # READING IN SAMPLE INFORMATION
    # THIS IS NEEDED FOR THE ALIGNED PCA TO WORK
    jdf = pd.read_csv('journal.tsv',sep='\t',index_col=0)
    jdf = jdf.loc[:,adf.columns.values]
    #
    alignment_label , sample_label = 'Disease' , None
    add_labels = ['Cell-line']
    #
    cmd                = 'max'
    # WRITE FILES AND MAKE NOISE
    bVerbose           = True
    # CREATE AN OPTIMIZED REPRESENTATION
    bExtreme           = True
    # WE MIGHT WANT SOME SPECIFIC INTERSECTIONS OF THE HIERARCHY
    n_clusters         = [20,40,60,80,100]
    # USE ALL INFORMATION
    n_components       = None
    umap_dimension     = 2
    n_neighbors        = 20
    local_connectivity = 20.
    transform_seed     = 42
    #
    print ( adf , jdf )
    #
    # distance_type = 'correlation,spearman,absolute' # DONT USE THIS
    distance_type = 'covariation' # BECOMES CO-EXPRESSION BASED
    #
    results = full_mapping ( adf , jdf                  ,
        bVerbose = bVerbose             ,
        bExtreme = bExtreme             ,
        n_clusters = n_clusters         ,
        n_components = n_components     ,
        distance_type = distance_type   ,
        umap_dimension = umap_dimension ,
        umap_n_neighbors = n_neighbors  ,
        umap_local_connectivity = local_connectivity ,
        umap_seed = transform_seed      ,
        hierarchy_cmd = cmd             ,
        add_labels = add_labels         ,
        alignment_label = alignment_label ,
        sample_label = None     )
    #
    map_analytes        = results[0]
    map_samples         = results[1]
    hierarchy_analytes  = results[2]
    hierarchy_samples   = results[3]
```
or just call it using the default values:
```
import pandas as pd
import numpy  as np

if __name__ == '__main__' :
    from biocarta.quantification import full_mapping
    #
    adf = pd.read_csv('analytes.tsv',sep='\t',index_col=0)
    #
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy()
    jdf = pd.read_csv('journal.tsv',sep='\t',index_col=0)
    jdf = jdf.loc[:,adf.columns.values]
    #
    alignment_label , sample_label = 'Disease' , None
    add_labels = ['Cell-line']
    #
    results = full_mapping ( adf , jdf  ,
        bVerbose = True			,
        n_clusters = [40,80,120]        ,
        add_labels = add_labels         ,
        alignment_label = alignment_label )
    #
    map_analytes        = results[0]
    map_samples         = results[1]
    hierarchy_analytes  = results[2]
    hierarchy_samples   = results[3]
```
and plotting the information of the map analytes yields :
[Cancer Disease Example](https://gist.github.com/rictjo/9cc40579914a51bffe7df442fec140f4)

You can also run an alternative algorithm where the UMAP coordinates are employed directly for clustering by setting
```
    results = full_mapping ( adf , jdf  ,
        bVerbose = True			        ,
        bUseUmap = True                 ,
        n_clusters = [40,80,120]        ,
        add_labels = add_labels         ,
        alignment_label = alignment_label )
```
with the following [results](https://gist.github.com/rictjo/8be5b5a9cc7f06ea7455d6c6ecc11ad8).

Download the zip and open the html index:
```
chromium index.html
```
