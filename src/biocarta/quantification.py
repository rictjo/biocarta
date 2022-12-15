"""
Copyright 2022 RICHARD TJÖRNHAMMAR
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import pandas as pd
import umap

def distance_calculation ( coordinates:np.array ,
                           distance_type:str , bRemoveCurse:bool=False ,
                           nRound:int = None ) -> np.array :
    # FROM IMPETUOUS-GFA
    crds = coordinates
    if 'correlation' in distance_type :
        from impetuous.quantification import spearmanrho,pearsonrho
        if 'pearson' in distance_type :
            corr =  pearsonrho( crds , crds )
        else :
            corr = spearmanrho( crds , crds )
        if 'absolute' in distance_type :
            corr = np.abs( corr )
        if 'square' in distance_type :
            corr = corr**2
        distm = 1 - corr
    else :
        from scipy.spatial.distance import pdist,squareform
        distm = squareform( pdist( crds , metric = distance_type ))
    if bRemoveCurse :
        from impetuous.reducer import remove_curse
        distm = remove_curse ( distm , nRound = nRound )
    return ( distm )


def cluster_appraisal( x:pd.Series , garbage_n = 0 ) :
    """
Clustering Optimisation Method for Highly Connected Data
Richard Tjörnhammar
https://arxiv.org/abs/2208.04720v2
    """
    # FROM IMPETUOUS-GFA
    from collections import Counter
    decomposition = [ tuple((item[1],item[0])) for item in Counter(x.values.tolist()).items() if item[1]
 > garbage_n ]
    N     = len ( x )
    A , B = 1 , N
    if len(decomposition) > 0 :
        A = np.mean(decomposition,0)[0]
        B = len(decomposition)
    else :
        decomposition = [ tuple( (0,0) ) ]
    decomposition .append( tuple( ( B ,-garbage_n )) )
    decomposition .append( tuple( ( A*B/(A+B) , None ) ) )
    return ( decomposition )

def generate_clustering_labels ( distm:np.array , cmd:str='max' ,
                            bExtreme:bool=False , n_clusters:int = None) :
    from impetuous.clustering import sclinkages
    res         = sclinkages( distm , cmd )['F']
    index       = list( res.keys() )
    if labels_f is None :
        labels_f = range(len(distm_features))
    hierarch_df = pd.DataFrame ( res.values() , index=list(res.keys()) , columns = labels_f )
    cluster_df  = res_df.T .apply( lambda x: cluster_appraisal(x,garbage_n = 0) )
    clabels_o , clabels_n = None , None
    if bExtreme :
        imax            = np.argmax( [ v[-1][0] for v in cluster_df.values ] )
        clabels_o       = hierarchi_df.iloc[imax,:].values
    if not n_clusters is None :
        jmax            = sorted ( [ i for i in range(len(df)) \
                            if np.abs( len( df.iloc[i].values )-2 - enforce_n_clusters ) < np.round(n_clusters*0.1) ])[0]
        clabels_n       = hierarch_df.iloc[jhit,:].values
    return ( clabels_n , clabels_o )


def create_mapping ( distm:np.array  , cmd:str = 'max' ,
                     n_clusters:int     = None  ,
                     bExtreme:bool      = False ,
                     bDoUmap:bool       = True  ,
                     umap_dimension:int = 2     ,
                     n_neighbors:int    = 20 ,
                     local_connectivity:float = 20 ,
                     transform_seed:int = 42 ) -> pd.DataFrame :

    clabels_o , clabels_n = generate_clustering_labels ( distm , cmd = cmd ,
                                 n_clusters = n_clusters ,
                                 bExtreme = bExtreme )
    u,s,vt = np.linalg.svd ( distm , False )
    Xf = u*s
    if bDoUmap :
        Uf = pd.DataFrame( umap.UMAP( local_connectivity      = local_connectivity ,
                                      n_components            = umap_dimension     ,
                                      n_neighbors             = n_neighbors        ,
                                      transform_seed          = transform_seed     ,
                                        ).fit_transform( Xf ) )
    resdf = pd.DataFrame( Xf ,
                  index      = adf.index.values ,
                  columns    = [ 'MFX.'+str(i) for i in range( len(Xf.T) ) ] )

    for i in range(umap_dimension) :
        resdf.loc[:,'UMAP.'+str(i)] = Uf.iloc[:,i].values
    if not clabels_o is None :
        resdf.loc['cids,max']  = clabels_o
    if not clabels_n is None :
        resdf.loc['cids.user'] = clabels_n
    return ( resdf )

if __name__ == '__main__' :
    #
    adf = pd.read_csv('analytes.tsv',sep='\t',index_col=0)
    labels_f = adf.index.values
    labels_s = adf.columns.values
    #
    print ( adf.apply(lambda x:np.sum(x)).values )
    #
    jdf = pd.read_csv('journal.tsv',sep='\t',index_col=0)
    alignment_label , sample_label = 'Cell-line' , 'sample'
    cmd = 'max'
    bVerbose = True
    bExtreme = False
    n_clusters = 80
    n_components = None
    bDoUmap = True
    umap_dimension = 2
    n_neighbors = 20
    local_connectivity = 20
    transform_seed = 42
    #
    print ( adf , jdf )
    #
    distance_type = 'correlation,spearman,squared'
    # distance_type = 'euclidean'
    #
    from impetuous.clustering import sclinkages
    distm_features = distance_calculation ( adf.values  , distance_type ,
                             bRemoveCurse = True , nRound = 4 )
    if bVerbose :
        print ( distm_features )
    #
    resdf = create_mapping ( distm = distm_features    , cmd = cmd ,
                     n_clusterst = n_clusters  , bExtreme = bExtreme ,
                     bDoUmap     = bDoUmap     , umap_dimension = umap_dimension,
                     n_neighbors = n_neighbors , local_connectivity = local_connectivity ,
                     transform_seed = transform_seed )

    resdf.to_csv('resdf.tsv',sep='\t')

    distm_samples  = distance_calculation ( adf.T.values, distance_type , bRemoveCurse = True )
    #
    if not jdf is None :
        if not ( sample_label is None or alignment_label is None ) :
            from impetuous.quantification import multivariate_aligned_pca
            pcas_df , pcaw_df = multivariate_aligned_pca ( analytes_df , journal_df ,
                    sample_label = sample_label , align_to = alignment_label ,
                    n_components = n_components )
    print ( pcas_df )
