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
import pandas as pd
import numpy as np
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
        corr = 0.5 * ( corr + corr.T ) # THIS SHOULD BE REMOVED ONCE DISC. LOCATED
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
#from impetuous.quantification import distance_calculation

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
#from impetuous.clustering import clustering_appraisal

def generate_clustering_labels ( distm:np.array , cmd:str='max' ,
                                 bExtreme:bool=False , n_clusters:int = None) -> tuple :
    # FROM IMPETUOUS-GFA
    from impetuous.clustering import sclinkages
    res         = sclinkages( distm , cmd )['F']
    index       = list( res.keys() )
    if labels_f is None :
        labels_f = range(len(distm_features))
    hierarch_df = pd.DataFrame ( res.values() , index=list(res.keys()) , columns = labels_f )
    cluster_df  = res_df.T .apply( lambda x: cluster_appraisal(x,garbage_n = 0) )
    clabels_o , clabels_n = None , None
    screening    = np.array( [ v[-1][0] for v in cluster_df.values ] )
    level_values = np.array( res.keys() )
    if bExtreme :
        imax            = np.argmax( screening )
        clabels_o       = hierarchi_df.iloc[imax,:].values.tolist()
    if not n_clusters is None :
        jmax            = sorted ( [ i for i in range(len(df)) \
                            if np.abs( len( df.iloc[i].values )-2 - enforce_n_clusters ) <= np.ceil(n_clusters*0.1) ])[0]
        clabels_n       = hierarch_df.iloc[jhit,:].values.tolist()
    return ( clabels_n , clabels_o , np.array( [level_values,screening] ) )
#from impetuous.clustering import generate_clustering_labels

def create_mapping ( distm:np.array , cmd:str = 'max' ,
                     index_labels:list[str] = None ,
                     n_clusters:int     = None  ,
                     bExtreme:bool      = True ,
                     bDoUmap:bool       = True  ,
                     umap_dimension:int = 2     ,
                     n_neighbors:int    = 20    ,
                     MF:tuple[np.array] = None  ,
                     local_connectivity:float = 20 ,
                     transform_seed:int = 42 ) -> pd.DataFrame :

    clabels_o , clabels_n , screening = generate_clustering_labels ( distm ,
            cmd = cmd , n_clusters = n_clusters ,
            bExtreme = bExtreme )
    if MX is None : # IF NOT PRECOMPUTED
        u , s , vt = np.linalg.svd ( distm , False )
        Xf = u*s # np.sqrt(s)
    else :
        Xf = MF[0]*MF[1]
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


def full_mapping ( adf:pd.DataFrame , jdf:pd.DataFrame ,
        bVerbose = False , bExtreme = True , force_n_clusters = None ,
        n_components = None , bDoUmap = True ,
        distance_type:str = 'correlation,spearman,absolute' ,
        umap_dimension:int = 2 , umap_n_neighbors:int = 20 , umap_local_connectivity:float = 20.,
        umap_seed = 42, hierarchy_cmd = 'max' , add_labels=None ) -> tuple[pd.DataFrame]:
    #
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy()
    jdf = jdf.loc[:,adf.columns.values.tolist() ].copy()
    #
    n_neighbors = umap_n_neighbors
    local_connectivity = umap_local_connectivity
    transform_seed = umap_seed
    cmd = hierarchy_cmd
    #
    from impetuous.clustering import sclinkages
    distm_features = distance_calculation ( adf.values  , distance_type ,
                             bRemoveCurse = True , nRound = 4 )
    if bVerbose :
        print ( distm_features )
    #
    resdf_f = create_mapping ( distm = distm_features ,
                     index_labels = labels_f , cmd = cmd ,
                     n_clusters  = n_clusters  , bExtreme = bExtreme ,
                     bDoUmap     = bDoUmap     , umap_dimension = umap_dimension,
                     n_neighbors = n_neighbors , local_connectivity = local_connectivity ,
                     transform_seed = transform_seed )
    #
    if bVerbose :
        print ( 'STORING RESULTS 1 > ', 'resdf_f.tsv' )
        resdf_f .to_csv( 'resdf_f.tsv',sep='\t' )
    #
    distm_samples  = distance_calculation ( adf.T.values, distance_type , bRemoveCurse = True )
    resdf_s = create_mapping ( distm = distm_samples ,
                     index_labels = labels_s    , cmd = cmd ,
                     n_clusters   = n_clusters  , bExtreme = bExtreme ,
                     bDoUmap      = bDoUmap     , umap_dimension = umap_dimension,
                     n_neighbors  = n_neighbors , local_connectivity = local_connectivity ,
                     transform_seed = transform_seed )
    if bVerbose :
        print ( 'STORING RESULTS 2 > ', 'resdf_s.tsv' )
        resdf_s .to_csv( 'resdf_s.tsv',sep='\t' )
    #
    pcas_df , pcaw_df = None , None
    if not jdf is None :
        if not ( sample_label is None or alignment_label is None ) :
            from impetuous.quantification import multivariate_aligned_pca
            pcas_df , pcaw_df = multivariate_aligned_pca ( adf , jdf ,
                    sample_label = sample_label , align_to = alignment_label ,
                    n_components = n_components , add_labels = add_labels )
    if bVerbose :
        print ( 'STORING RESULTS 3,4 > ', 'pcas_df.tsv', 'pcaw_df.tsv' )
        pcas_df .to_csv( 'pcas_df.tsv','\t' )
        pcaw_df .to_csv( 'pcaw_df.tsv','\t' )

    return ( resdf_f , pcas_df , resdf_s , pcaw_df )


if __name__ == '__main__' :
    #
    adf = pd.read_csv('analytes.tsv',sep='\t',index_col=0)
    labels_f = adf.index.values.tolist()
    labels_s = adf.columns.values.tolist()
    #
    distance_type = 'correlation,spearman,squared'
    distm_features = distance_calculation ( adf.values  , distance_type ,
                             bRemoveCurse = True , nRound = 4 )
    #
    print ( np.sum( distm_features,0 ) )
    print ( np.sum( distm_features,1 ) )
    print ( 1./np.std( distm_features,0 ) )
    print ( 1./np.std( distm_features,1 ) )
    #
    print ( adf.apply(lambda x:np.sum(x)).values )
    #
    jdf = pd.read_csv('journal.tsv',sep='\t',index_col=0)
    alignment_label , sample_label = 'Cell-line' , 'sample'
    add_labels = ['Disease']
    #
    cmd                = 'max'
    bVerbose           = True
    bExtreme           = True
    n_clusters         = 80
    n_components       = None
    bDoUmap            = True
    umap_dimension     = 2
    n_neighbors        = 20
    local_connectivity = 20
    transform_seed     = 42
    #
    print ( adf , jdf )
    #
    distance_type = 'correlation,spearman,squared'
    # distance_type = 'euclidean'
    #
    full_mapping ( adf , jdf ,
        bVerbose = bVerbose , bExtreme = bExtreme , force_n_clusters = n_clusters ,
        n_components = n_components , bDoUmap = bDoUmap ,
        distance_type = distance_type  ,
        umap_dimension = umap_dimension ,
        umap_n_neighbors = n_neighbors  ,
        umap_local_connectivity = local_connectivity ,
        umap_seed = transform_seed , hierarchy_cmd = 'max',
        add_labels = add_labels )


