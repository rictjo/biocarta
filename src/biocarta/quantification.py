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

from impetuous.quantification	import distance_calculation
from impetuous.clustering	import generate_clustering_labels
from impetuous.clustering	import distance_matrix_to_absolute_coordinates

def create_mapping ( distm:np.array , cmd:str = 'max'	,
                     index_labels:list[str]	= None	,
                     n_clusters:int		= None	,
                     bExtreme:bool		= True	,
                     bDoUmap:bool		= True	,
                     umap_dimension:int		= 2	,
                     n_neighbors:int		= 20	,
                     MF:np.array		= None	,
                     n_proj:int			= 2	,
                     local_connectivity:float	= 20	,
                     transform_seed:int = 42 ) -> tuple[pd.DataFrame] :

    clabels_n , clabels_o , hierarch_df, sol = generate_clustering_labels ( distm ,
            labels = index_labels , cmd = cmd , n_clusters = n_clusters ,
            bExtreme = bExtreme )

    if n_proj is None :
        n_proj = len(distm)

    if MF is None : # IF NOT PRECOMPUTED
        Xf = distance_matrix_to_absolute_coordinates ( distm ,
			n_dimensions = -1 , bLegacy = False )
    else :
        Xf = MF
    #
    print ( Xf , np.shape( Xf ) )
    if bDoUmap :
        Uf = pd.DataFrame( umap.UMAP( local_connectivity	= local_connectivity	,
                                      n_components		= int(umap_dimension)	,
                                      n_neighbors		= int(n_neighbors)	,
                                      transform_seed		= transform_seed	,
                                        )\
					.fit_transform( Xf ) )
					#.fit_transform( distm ) )
    #
    print ( Uf )
    resdf = pd.DataFrame( np.array([ Xf.T[i] for i in range( len(Xf.T) ) if i<n_proj ]).T  ,
                  index      = index_labels ,
                  columns    = [ 'MFX.'+str(i) for i in range( len(Xf.T) ) if i<n_proj ] )

    for i in range(umap_dimension) :
        resdf.loc[:,'UMAP.'+str(i)] = Uf.iloc[:,i].values
    if not clabels_o is None :
        resdf.loc[:,'cids.max']  = clabels_o
    if not clabels_n is None :
        resdf.loc[:,'cids.user'] = clabels_n
    return ( resdf, hierarch_df , pd.DataFrame(sol) )

def full_mapping ( adf:pd.DataFrame , jdf:pd.DataFrame ,
        bVerbose:bool = False , bExtreme:bool = True , force_n_clusters:int = None ,
        n_components:int = None , bDoUmap:bool = True ,
        distance_type:str = 'correlation,spearman,absolute' ,
        umap_dimension:int = 2 , umap_n_neighbors:int = 20 , umap_local_connectivity:float = 20. ,
        umap_seed:int = 42 , hierarchy_cmd:str = 'max' ,
        add_labels:list[str] = None, n_projections:int = 2,
        directory:str = None ) -> tuple[pd.DataFrame] :
    #
    if bVerbose :
        import time
        header_str = 'YMDHMS_'+'_'.join( list( str(t) for t in time.gmtime())[:-3] )+'_'
        if not directory is None :
            if not directory[-1] == '/' :
                directory = directory + '/'
            header_str = directory + header_str
    #
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy()
    jdf = jdf .loc[ :,adf.columns.values.tolist() ].copy()
    #
    n_neighbors		= umap_n_neighbors
    local_connectivity	= umap_local_connectivity
    transform_seed	= umap_seed
    cmd			= hierarchy_cmd
    bRemoveCurse_	= True
    nRound_		= None
    #
    from impetuous.special import zvals
    input_values = zvals( adf.values )['z']
    #
    MF_f , MF_s = None , None
    if False : # DECOMPOSE THE DISTM IF FALSE, NOT DONE
        u , s , vt = np.linalg.svd ( input_values , False )
        MF_f = u*s
        MF_s = vt.T*s

    from impetuous.clustering import sclinkages
    distm_features = distance_calculation ( input_values  , distance_type ,
                             bRemoveCurse = bRemoveCurse_ , nRound = nRound_ )

    divergence = lambda r : np.exp(r)
    distm_features *= divergence ( distm_features )

    resdf_f , hierarch_f_df , soldf_f = create_mapping ( distm = distm_features ,
                     index_labels = adf.index.values , cmd = hierarchy_cmd , MF = MF_f ,
                     n_clusters  = n_clusters  , bExtreme = bExtreme ,
                     bDoUmap     = bDoUmap     , umap_dimension = umap_dimension,
                     n_neighbors = n_neighbors , local_connectivity = local_connectivity ,
                     transform_seed = transform_seed, n_proj = n_projections )

    if bVerbose :
        print ( 'STORING RESULTS > ', 'resdf_f.tsv , soldf_f.tsv , hierarch_f.tsv' )
        resdf_f .to_csv( header_str + 'resdf_f.tsv' , sep='\t' )
        soldf_f .to_csv( header_str + 'soldf_f.tsv' , sep='\t' )
        hierarch_f_df.to_csv( header_str + 'hierarch_f.tsv' , sep='\t' )
    #
    distm_samples  = distance_calculation ( input_values.T , distance_type ,
			     bRemoveCurse = bRemoveCurse_ , nRound = nRound_ )
    distm_samples *= divergence ( distm_samples )

    resdf_s , hierarch_s_df , soldf_s = create_mapping ( distm = distm_samples ,
                     index_labels = adf.columns.values , cmd = hierarchy_cmd , MF = MF_s ,
                     n_clusters   = n_clusters  , bExtreme = bExtreme ,
                     bDoUmap      = bDoUmap     , umap_dimension = umap_dimension,
                     n_neighbors  = n_neighbors , local_connectivity = local_connectivity ,
                     transform_seed = transform_seed , n_proj = n_projections )
    #
    if bVerbose :
        print ( 'STORING RESULTS > ', 'resdf_s.tsv , soldf_s.tsv , hierarch_s.tsv' )
        resdf_s .to_csv( header_str + 'resdf_s.tsv',sep='\t' )
        soldf_s .to_csv( header_str + 'soldf_s.tsv',sep='\t' )
        hierarch_s_df.to_csv( header_str + 'hierarch_s.tsv' , sep='\t' )
    #
    pcas_df , pcaw_df = None , None
    if not jdf is None :
        if not ( sample_label is None or alignment_label is None ) :
            from impetuous.quantification import multivariate_aligned_pca
            pcas_df , pcaw_df = multivariate_aligned_pca ( adf , jdf ,
                    sample_label = sample_label , align_to = alignment_label ,
                    n_components = n_components , add_labels = add_labels )
            if bVerbose :
                print ( 'STORING RESULTS > ', 'pcas_df.tsv', 'pcaw_df.tsv' )
                pcas_df .to_csv( header_str + 'pcas_df.tsv','\t' )
                pcaw_df .to_csv( header_str + 'pcaw_df.tsv','\t' )

            resdf_f = pd.concat( [resdf_f.T, pcas_df.T] ).T
            resdf_s = pd.concat( [resdf_s.T, pcaw_df.T] ).T
    #
    if bVerbose:
        print ( 'RETURNING: ')
        print ( 'FEATURE MAP, SAMPLE MAP, FULL FEATURE HIERARCHY, FULL SAMPLE HIERARCHY' )
    return ( resdf_f , resdf_s , hierarch_f_df, hierarch_s_df )


if __name__ == '__main__' :
    #
    adf = pd.read_csv('analytes.tsv',sep='\t',index_col=0)
    #
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy()
    #
    jdf = pd.read_csv('journal.tsv',sep='\t',index_col=0)
    jdf = jdf.loc[:,adf.columns.values]
    #
    # adf=adf.iloc[:200,:]
    alignment_label , sample_label = 'Disease' , 'sample'
    add_labels = ['Cell-line']
    #
    cmd                = 'max'
    bVerbose           = True
    bExtreme           = True
    n_clusters         = 80
    n_components       = None
    bDoUmap            = True
    umap_dimension     = 2
    n_neighbors        = 20.
    local_connectivity = 20.
    transform_seed     = 42
    #
    print ( adf , jdf )
    #
    distance_type = 'correlation,spearman,absolute'
    # distance_type = 'euclidean'
    #
    print ( 'DOING THIS' , bVerbose )
    full_mapping ( adf , jdf ,
        bVerbose = bVerbose	,
	bExtreme = bExtreme	,
	force_n_clusters = n_clusters	,
        n_components = n_components 	,
	bDoUmap = bDoUmap	,
        distance_type = distance_type  	,
        umap_dimension = umap_dimension	,
        umap_n_neighbors = n_neighbors	,
        umap_local_connectivity = local_connectivity ,
        umap_seed = transform_seed	,
	hierarchy_cmd = cmd 	,
        add_labels = add_labels )


