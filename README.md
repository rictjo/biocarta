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
with the following [results](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/8be5b5a9cc7f06ea7455d6c6ecc11ad8/raw/e00ea663a1218718f542744a939e0b05c604e8ab/index.html).

Download the zip and open the html index:
```
chromium index.html
```

# Other generated solutions

The clustering visualisations were created using the [Biocarta](https://pypi.org/project/biocarta/) and [hvplot](https://pypi.org/project/hvplot/) :

What groupings corresponds to biomarker variance that describe them? Here are two visualisations of that:

Diseases :
[cancers](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/feaa14d3c83cb84ebc23b89662e0702c/raw/19032d8653e35f0be32212ea73ae57a69d50004c/index.html)


Tissues :
[tissues](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/5e760b8c4fd3da4842813a4a0cea422c/raw/caa18f0391dc389fb8fc56ae8ac2bc4f7046a939/index.html)


