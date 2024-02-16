# Biocartograph
Creating Cartographic Representations of Biological Data
[![DOI](https://zenodo.org/badge/578172132.svg)](https://zenodo.org/badge/latestdoi/578172132)

# Installation
```
pip install biocartograph
```
You can also build a [nix environment](https://github.com/rictjo/versioned-nix-environments/blob/main/env/versioned_R_and_Python.nix) for code execution if you have installed the [nix package](https://nixos.org/download.html) manager. You can enter it via a terminal by issuing:
```
nix-shell versioned_R_and_Python.nix
```

# Example code
We generally work with short, or compact, format data frames. One describing the analytes (often abbreviated "adf") :

|NAME       |NGT_mm12_10591 | ... | DM2_mm81_10199 |
|:---       |           ---:|:---:|            ---:|
|215538_at  |    16.826041 | ... | 31.764484       |
|...        |              |     |                 |
|LDLR       |   19.261185  | ... | 30.004612       |

and one journal describing the sample metadata (often abbreviated "jdf") :

|      |NGT_mm12_10591 | ... | DM2_mm81_10199 |
|:---    |         ---:|:---:|            ---:|
| Disease  |  Prostate Cancer  | ... | Gastric Cancer |
| Cell-line| 143B   | ... | 22Rv1 |
| Tissue |  Prostate | ... | Gastric Urinary tract |

if these are stored as tab-delimited text files then it is straightforward to read them in from disc.
```
if __name__ == '__main__' :
    from biocartograph.quantification import full_mapping
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
```

Next, we specify how to conduct the calculation
```
    consensus_labels = ['Tissue']
    results = full_mapping ( adf , jdf                                  ,
            bVerbose                    = True                          ,
            alignment_label             = alignment_label               ,
            umap_n_neighbors            = 20                            ,
            umap_local_connectivity     = 20.                           ,
            bUseUmap                    = False                         ,
            consensus_labels            = consensus_labels              ,
            distance_type               = 'coexpression'                ,
            hierarchy_cmd               = 'ward' ,
            directory                   = '../results' ,
            n_clusters                  = sorted([ 10 , 20 , 30 , 40 , 60 , 70 , 90 , 80 , 100 ,
                                                120 , 140 , 160 , 180 , 200 ,
                                                250 , 300 , 350 , 400 , 450 , 500 ,
                                                600 , 700 , 800 , 900 , 1000 ])  )
    #
    map_analytes        = results[0]
    map_samples         = results[1]
    hierarchy_analytes  = results[2]
    hierarchy_samples   = results[3]
    header_str = results[0].index.name
```
In this example, we didn't calculate any projection properties relating to the Cell-line label. We also decided on outputting some specific cuts through the hierarchical clustering solution corresponding to different amounts of clusters. We generate multivariate projected PCA files for all the consensus and alignment labels. Plotting the information on the map analytes PCA projections yields:
[Cancer Disease mPCA Example](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/ed1d6768f9ffa11ada2a6c5ad75094ca/raw/74dd22f261f9925fd005d539a851ae013df0c574/index.html)

You can also run an alternative algorithm where the UMAP coordinates are employed directly for clustering by setting `bUseUmap=True` with the following [results](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/8be5b5a9cc7f06ea7455d6c6ecc11ad8/raw/e00ea663a1218718f542744a939e0b05c604e8ab/index.html), or download the gist zip and open the html index:
```
chromium index.html
```

# Other generated solutions

The clustering visualisations were created using the [Biocartograph](https://pypi.org/project/biocartograph/) and [hvplot](https://pypi.org/project/hvplot/) :

What groupings correspond to biomarker variance that describes them? Here are some visualisations of that:

[Cell-line Diseases](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/786fa0b8706bb02ccb2cd3bba3f5c35d/raw/0ad4261d267f7550ddae1714e16aec8301127af0/index.html)
[Tissues](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/f8556c6d5b9f0369138eb194356e2818/raw/cd42c623460f049b39fbc311253526bd0fec0cd5/index.html)
[Single cells](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/e7a7c0218e053ec0b631b17a52eb23fc/raw/b5c5b3249e3a0b9698231d424986072e37379ea3/index.html)
[Brain tissues](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/cb0d68fb6a853b862806b6ba1cd01ea4/raw/48d1441df9299dd67774529f67e230128603db3e/index.html)
[Blood immune cells](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/c55222e45b04b35c48f4506a0d463ba9/raw/885f089ca81d3f8705b70c9666ab168dfcc01188/index.html)

We can also make more [elaborate visualisation](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/0609bfb6a6113703155f3c08058d1855/raw/47b27519da996cde17f32e7d386ea527a89eba8a/index.html) applications with the information that the biocartograph calculates.

## Mammals 
[interactive rat](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/ceec25278234be1079a055eb77588eea/raw/a306bc5fa1f227b0ecabe0f46577e5646c05f8b2/index.html)
[interactive pig](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/b6af5a52d1aee67ea9d84d9edc2af169/raw/388900e9518c3ee869006ee032bfb5ef9a26406b/index.html)

# Enrichment results
If we have gmt files describing what groups of our analytes might be in then we can calculate enrichment properties for gene groupings (clusters). One resource for obtaining information is the [Reactome](https://reactome.org/download-data) database. If the pathway definitions are hierarchical then you can also supply the parent-child list and calculate treemap enrichments for all your clusters.
[Example of biocartograph treemap cluster](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/146ba66109c6554684dc387348d21a82/raw/a32f1e7c80cc6ebe53c33039e2adfb4512e3ce4b/index.html)

The code for doing it might look something like this :
```
    from biocartograph.special import generate_atlas_files
    import biocartograph.enrichment as bEnriched

    df_ = pd.read_csv( header_str + 'resdf_f.tsv' , index_col=0 , sep='\t' )
    df_ .loc[:,'cids.max' ]     = [ str(v) for v in df_.loc[:,'cids.max' ].values       ] # OPTIMAL SOLUTION
    enr_dict = bEnriched.calculate_for_cluster_groups ( df_ , label = 'cids.max' ,
                    gmtfile = '../data/Reactome/reactome_v71.gmt' , pcfile = '../data/Reactome/NewestReactomeNodeRelations.txt' ,
                    group_identifier = 'R-HSA' , significance_level = 0.1 )
    for item in enr_dict.items() :
        item[1].to_csv( header_str + 'treemap_c' + str(item[0])+'.tsv',sep='\t' )
```
You can also produce a `gmt` and `pcfile` of your own from the clustering solution labels:
```
    from biocartograph.special import generate_atlas_files , reformat_results_and_print_gmtfile_pcfile
    cl_gmtname , cl_pcname = reformat_results_and_print_gmtfile_pcfile ( header_str , hierarchy_id = 'cids.max', hierarchy_level_label = 'HCLN' )
```
For group factor enrichments simply use the `bEnriched.from_multivariate_group_factors` method instead. This will produce results that can be visualised like this:
[biocartograph gfa Reactome enrichment](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/42ec85df088a0c40de339a78322594bd/raw/0725bea467b0c153298655e3a0555670a812e80f/index.html) or the [cluster label gfa enrichments](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/5d83a85537839232f34edccde1cdc8e6/raw/40c49013a55213405a6b6609f9ab31c883668d5d/index.html)

[cluster treemap svg](https://gist.github.com/rictjo/26192142e3d58c4849cacf96f1a87235) using the biocartograph.special utilities

# Creating a nested file structure
There is a function within the `biocartograph` package that can be used to package your generated results into a more easily parsed directory. This function can be called via :
```
generate_atlas_files ( header_str )
```
This will produce cluster annotation information taken from the enrichment files as well as the sample labels used.

[Cell-line Disases](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/a1224cea00421079992558e58642f2c5/raw/602bf63c9baa9ba5931f93423237ad5189c642b6/index.html)
[Tissues](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/09cb0ea4ef04fd8f28b26eb8e15a02db/raw/92264c50a9f17e4771832710d92638fc5b5d6437/index.html) 
[Single cells](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/f2f6aeee8574bd39fd5b5a7c3f9fd8bd/raw/bd10e243a3899e0f6e9fc7e80c2669ed1b13d299/index.html)
[Brain tissues](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/cc17695a59598c0cdac470b71a08d28f/raw/b73319f437f43490bc1efd71798e86545b9680bc/index.html)
[Blood immune cells](https://rictjo.github.io/?https://gist.githubusercontent.com/rictjo/af637bdec9feb8e37b1d4ea8909f6258/raw/b83108313262d0c4ccef4e7771116e4cfe7566af/index.html)

# Upcoming
Hopefully, an even more helpful wiki will be provided in the future.
