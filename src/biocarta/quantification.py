"""
Copyright 2023 RICHARD TJÖRNHAMMAR
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

import impetuous.quantification as impq
import impetuous.clustering     as impc

from impetuous.quantification	import distance_calculation
from impetuous.clustering	import generate_clustering_labels
from impetuous.clustering	import distance_matrix_to_absolute_coordinates

def create_mapping ( distm:np.array , cmd:str	= 'max'	,
                     index_labels:list[str]	= None	,
                     n_clusters:list[int]	= None	,
                     bExtreme:bool		= True	,
                     bUseUmap:bool		= False	,
                     umap_dimension:int		= 2	,
                     n_neighbors:int		= 20	,
                     MF:np.array		= None	,
                     bNonEuclideanBackprojection:bool = False ,
                     n_proj:int			= 2	,
                     local_connectivity:float	= 20.	,
                     Sfunc = lambda x:np.mean(x,0) ,
                     transform_seed:int = 42 ) -> tuple[pd.DataFrame] :
    #
    if MF is None :
        # IF NOT PRECOMPUTED
        # WASTE OF COMPUTATION ...
        if bNonEuclideanBackprojection :
            u , s , vt = np.linalg.svd ( distm , False )
            Xf = u*s
        else :
            Xf = distance_matrix_to_absolute_coordinates ( distm ,
                n_dimensions = -1 , bLegacy = False )
        MF = Xf
    else :
        Xf = MF
    #
    Uf = pd.DataFrame( umap.UMAP( local_connectivity	= local_connectivity	,
                                  n_components		= int(umap_dimension)	,
                                  n_neighbors		= int(n_neighbors)	,
                                  transform_seed	= transform_seed	,
                                    ) .fit_transform( Xf ) )
    resdf = pd.DataFrame( Uf )
    resdf .index = index_labels
    resdf .columns = ['UMAP.'+str(i) for i in range(umap_dimension)]
    #
    if bUseUmap :
        distm = distance_calculation ( Uf , 'euclidean' ,
                             bRemoveCurse = True , nRound = None )
    if not n_clusters is None :
        n_cl = n_clusters[0]
    else :
        n_cl = 1
    #
    clabels_n , clabels_o , hierarch_df, sol = generate_clustering_labels ( distm ,
            labels = index_labels , cmd = cmd , n_clusters = n_cl ,
            bExtreme = bExtreme , Sfunc=Sfunc )

    fully_connected_at          = float( hierarch_df.index.values[-1] )
    optimally_connected_at      = float( hierarch_df.index.values[np.argmax(sol[1])] )
    if n_proj is None :
        n_proj = len ( distm )
    if n_proj == -1 :
        n_proj = len(MF.T)
    resdf = pd.concat([resdf ,
		       pd.DataFrame( np.array( [mf for mf in MF.T][:n_proj] ).T ,
				columns= [ 'MFX.' + str(i) for i in range(n_proj)],index=resdf.index )],
		       axis=1 )
    #
    if not clabels_o is None :
        resdf.loc[:,'cids.max']  = clabels_o
    if not clabels_n is None :
        resdf.loc[:,'cids.user'] = clabels_n
        if 'list' in str(type(n_clusters)) :
            for i in range(len(hierarch_df)) :
                labs	= hierarch_df.iloc[i,:].values
                N	= len( set( labs ) )
                if N in set( n_clusters ) :
                    resdf.loc[:,'cids.user.'+str(N)] = labs
    #
    return ( resdf, hierarch_df , pd.DataFrame(sol) )


def full_mapping ( adf:pd.DataFrame , jdf:pd.DataFrame ,
        bVerbose:bool = True , bExtreme:bool = True , n_clusters:list[int] = None ,
        n_components:int = None , bUseUmap:bool = False , bPreCompute:bool=True ,
        distance_type:str  = 'covariation' , # 'correlation,spearman,absolute' ,
        umap_dimension:int = 2 , umap_n_neighbors:int = 20 , umap_local_connectivity:float = 1. ,
        umap_seed:int = 42 , hierarchy_cmd:str = 'max' , divergence = lambda r : np.exp(r) ,
        add_labels:list[str] = None , sample_label:str = None , alignment_label:str = None , bRemoveCurse:bool=False ,
        n_projections:int = 2 , directory:str = './' , bQN:int = None ,
        nNeighborFilter:list[int] = None , heal_symmetry_break_method:str = 'average' ,
        epls_ownership:str = 'angle' , bNonEuclideanBackprojection:bool = False ,
        Sfunc = lambda x:np.mean(x,0) , bAddPies:bool=False ) -> tuple[pd.DataFrame] :
    #
    import biocarta.special as biox
    #
    if bVerbose :
        print ( "TO DISABLE WRITING OF RESULTS TO", directory )
        print ( "SET THE directory=None " )
        print ( "TO MAKE BIOCARTA QUIET SET bVerbose=False" )
        import time
        header_str = 'YMDHMS_' + '_'.join( list( str(t) for t in time.gmtime())[:-3] )+'_'
        header_str = 'DMHMSY_' + time.ctime().replace(':','_').replace(' ','_') + '_'
        if not directory is None :
            if not directory[-1] == '/' :
                directory = directory + '/'
            header_str = directory + header_str
            #
            # HERE WE INCLUDE ALL THE RUN PARAMETERS THAT ARE ACCESIBLE AS INPUTS
            # AND FROM THE ENVIRONMENT AT THE START OF THE RUN
            #
            runinfo_file = 'params.txt'
            run_dict = { 'bVerbose:bool':bVerbose , 'bExtreme:bool':bExtreme , 'n_clusters:list[int]':n_clusters ,
        'n_components:int':n_components , 'bUseUmap:bool':bUseUmap , 'bPreCompute:bool':bPreCompute , 'distance_type:str':distance_type ,
        'umap_dimension:int':umap_dimension , 'umap_n_neighbors:int':umap_n_neighbors , 'umap_local_connectivity:float':umap_local_connectivity ,
        'umap_seed:int':umap_seed , 'hierarchy_cmd:str':hierarchy_cmd , 'divergence:lambda function': divergence ,
        'add_labels:list[str]':add_labels , 'sample_label:str':sample_label , 'alignment_label:str':alignment_label ,
        'bRemoveCurse:bool':bRemoveCurse , 'n_projections:int':n_projections , 'directory:str':directory , 'bQN:int':bQN ,
        'nNeighborFilter:list[int]':nNeighborFilter , 'heal_symmetry_break_method:str':heal_symmetry_break_method ,
        'epls_ownership:str':epls_ownership , 'bNonEuclideanBackprojection:bool':bNonEuclideanBackprojection ,
        'Sfunc':Sfunc , 'bAddPies:bool':bAddPies }
            ofile = open ( header_str + runinfo_file , 'w' )
            for item in run_dict.items():
                if 'list' in str(type(item[1])):
                    print ( item[0],'\t=\t [', ','.join([str(i) for i in item[1]]) ,']', file=ofile )
                else :
                    print ( item[0],'\t=\t [', str(item[1]) ,']', file=ofile )
            print ( 'PYTHON GLOBALS:\n ',
			'\n'.join([ '\t=\t '.join([str(i) for i in item if not i is None]) for item in globals().items() if not item is None ] ),
			file = ofile )
    #
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy().apply(pd.to_numeric)
    #
    comp_df = None
    if not jdf is None :
        if not ( alignment_label is None ) :
            if bVerbose :
                print ( "CONDUCTING COMPOSITIONAL ANALYSIS" )
            comp_df = biox.calculate_compositions ( adf , jdf, label = alignment_label , bAddPies=bAddPies )
            comp_df.columns = [ alignment_label +'.'+ c for c in comp_df.columns.values ]
            if bVerbose :
                print (  'FINISHED RESULTS > ' , 'composition.tsv' )
            if not directory is None:
                comp_df .to_csv (  header_str + 'composition.tsv' , sep='\t' )
    #
    if not bQN is None :
        adf = biox.quantile_class_normalisation ( adf , axis=bQN )
    jdf = jdf.loc [ :,adf.columns.values.tolist() ].copy()
    #
    n_neighbors		= umap_n_neighbors
    local_connectivity	= umap_local_connectivity
    transform_seed	= umap_seed
    cmd			= hierarchy_cmd
    bRemoveCurse_	= bRemoveCurse
    nRound_		= None
    if not n_components is None :
        m = np.min(np.shape(adf.values))
        if n_components >= m :
            n_components = m - 1
            print ( 'WARNING SETTING n_components TO :' , n_components )
    #
    from impetuous.special import zvals
    input_values = zvals( adf.values )['z']
    #
    input_values_f = input_values
    input_values_s = input_values.T
    MF_f , MF_s = None , None
    if 'covariation' in distance_type or 'coexpression' in distance_type or bPreCompute :
        u , s , vt = np.linalg.svd ( input_values , False )
        # SINCE INPUT IS MEAN CENTERED THE COMPONENT COORDINATES CORRESPOND TO THE COVARIATION MATRIX
        if not n_components is None :
            s[n_components:] *= 0
        MF_f = u*s	# EQUIV TO : np.dot(u,np.diag(s))
        MF_s = vt.T*s
        if 'absolute' in distance_type.split('secondary[')[0] :
            MF_f = np.abs( MF_f )
        if 'absolute' in distance_type.split('secondary[')[0] :
            MF_s = np.abs( MF_s )
    #
    if 'covariation' in distance_type or 'coexpression' in distance_type :
        input_values_f = MF_f
        input_values_s = MF_s
        if 'secondary[' in distance_type :	# THIS IS ACTUALLY A RATHER ODD THING TO DO
						# BUT SOME POEPLE WILL WANT TO DO IT ANYWAY
            distance_type = distance_type.split('secondary[')[1].split(']')[0]
        else :
            # HERE THE COMPONENTS ARE THE ABSOLUTE COORDINATES OF THE COVARIATION MATRIX
            # I.E. COV = np.dot( MF_s , MF_s.T )/( len(MF_s)-1 )
            distance_type = 'euclidean'
    else :
        if bVerbose :
            print ( "NOTIFICATION    >  I SUGGEST USING THE covariation SETTING FOR DISTANCE TYPE" )
    #
    distm_features = distance_calculation ( input_values_f , distance_type ,
                         bRemoveCurse = bRemoveCurse_ , nRound = nRound_ )
    #
    if not nNeighborFilter is None :
        # LEAVE THIS OUT IT IS CRAP
        print ( 'WARNING : CREATING SYMMETRY BREAK' )
        from impetuous.clustering import nearest_neighbor_graph_matrix
        snn_graph , gc_val = nearest_neighbor_graph_matrix ( distm=distm_features, nn=nNeighborFilter[0] )
        distm_features = distm_features * ( snn_graph <= gc_val ) # APPLYING SNN FILTER
        # FOR AGGLOMERATIVE HIERARCHICAL CLUSTERING THE MATRIX MUST BE SYMMETRIC
        distm_features = biox.symmetrize_broken_symmetry ( distm_features , method = heal_symmetry_break_method )
    if not bRemoveCurse_ :
        divergence  = lambda r : 1
    distm_features *= divergence ( distm_features )
    #
    resdf_f , hierarch_f_df , soldf_f = create_mapping ( distm = distm_features ,
                     index_labels = adf.index.values , cmd = hierarchy_cmd , MF = MF_f ,
                     n_clusters  = n_clusters  , bExtreme = bExtreme ,
                     bUseUmap    = bUseUmap    , umap_dimension = umap_dimension,
                     n_neighbors = n_neighbors , local_connectivity = local_connectivity ,
                     transform_seed = transform_seed, n_proj = n_projections ,
                     bNonEuclideanBackprojection = bNonEuclideanBackprojection ,
                     Sfunc = Sfunc )
    if bVerbose :
        print ( 'FINISHED RESULTS > ', 'resdf_f.tsv , soldf_f.tsv , hierarch_f.tsv' )
    if not directory is None:
        resdf_f .index.name = header_str
        resdf_f .to_csv( header_str + 'resdf_f.tsv' , sep='\t' )
        soldf_f .to_csv( header_str + 'soldf_f.tsv' , sep='\t' )
        hierarch_f_df .to_csv( header_str + 'hierarch_f.tsv' , sep='\t' )
    #
    distm_samples = distance_calculation ( input_values_s , distance_type ,
                             bRemoveCurse = bRemoveCurse_ , nRound = nRound_ )
    #
    if not nNeighborFilter is None : # LEAVE THIS OUT IT IS CRAP
        print ( 'WARNING : CREATING SYMMETRY BREAK' )
        snn_graph , gc_val = nearest_neighbor_graph_matrix ( distm=distm_samples, nn=nNeighborFilter[-1] )
        distm_samples = distm_samples * ( snn_graph <= gc_val )
        # FOR HIERARCHICAL CLUSTERING THE MATRIX MUST BE SYMMETRIC
        distm_samples = biox.symmetrize_broken_symmetry ( distm_samples , method = heal_symmetry_break_method )
    #
    distm_samples *= divergence ( distm_samples )
    #
    resdf_s , hierarch_s_df , soldf_s = create_mapping ( distm = distm_samples ,
                     index_labels = adf.columns.values	, cmd = hierarchy_cmd , MF = MF_s ,
                     n_clusters   = n_clusters  	, bExtreme = bExtreme ,
                     bUseUmap     = bUseUmap    	, umap_dimension = umap_dimension,
                     n_neighbors  = n_neighbors 	, local_connectivity = local_connectivity ,
                     transform_seed = transform_seed	, n_proj = n_projections ,
                     bNonEuclideanBackprojection = bNonEuclideanBackprojection ,
                     Sfunc = Sfunc )
    #
    if bVerbose :
        print ( 'FINISHED RESULTS > ', 'resdf_s.tsv , soldf_s.tsv , hierarch_s.tsv' )
    if not directory is None :
        resdf_s .to_csv( header_str + 'resdf_s.tsv',sep='\t' )
        soldf_s .to_csv( header_str + 'soldf_s.tsv',sep='\t' )
        hierarch_s_df.to_csv( header_str + 'hierarch_s.tsv' , sep='\t' )
    #
    pcas_df , pcaw_df = None , None
    if not jdf is None :
        if not ( alignment_label is None ) :
            if bVerbose :
                print ( "MULTIVAR ALIGNED PCA" )
            from impetuous.quantification import multivariate_aligned_pca
            if sample_label is None :
                jdf.loc['samplenames'] = jdf.columns.values
                sample_label = 'samplenames'
            pcas_df , pcaw_df = multivariate_aligned_pca ( adf , jdf ,
                    sample_label = sample_label , align_to = alignment_label ,
                    n_components = n_components , add_labels = add_labels )
            if bVerbose :
                print ( "ENCODED PLS REGRESSION" )
            from impetuous.quantification import run_rpls_regression as epls
            jdf = jdf.rename(index={alignment_label:'AL0xXx'})
            res = epls ( analyte_df=adf, journal_df=jdf, formula = 'Expression~C(AL0xXx)' , owner_by = epls_ownership )
            jdf = jdf.rename(index={'AL0xXx':alignment_label})
            res[0].columns = [ 'EPLS.' + v for v in res[0].columns.values ]
            res[1].columns = [ 'EPLS.' + v for v in res[1].columns.values ]
            if bVerbose :
                print ( 'FINISHED RESULTS > ', 'pcas_df.tsv', 'pcaw_df.tsv', 'epls_f.tsv' , 'epls_s.tsv' )
            if not directory is None:
                pcas_df .to_csv ( header_str + 'pcas_df.tsv', sep='\t' )
                pcaw_df .to_csv ( header_str + 'pcaw_df.tsv', sep='\t' )
                res[ 0 ].to_csv ( header_str + 'epls_f.tsv' , sep='\t' )
                res[ 1 ].to_csv ( header_str + 'epls_s.tsv' , sep='\t' )
            resdf_f = pd.concat( [resdf_f.T, pcas_df.T , res[0].T , comp_df.T ] ).T
            resdf_s = pd.concat( [resdf_s.T, pcaw_df.T , res[1].T ] ).T
    #
    if bVerbose :
        print ( 'RETURNING: ')
        print ( 'FEATURE MAP, SAMPLE MAP, FULL FEATURE HIERARCHY, FULL SAMPLE HIERARCHY' )
    if not header_str is None :
        resdf_f .index .name = header_str
    return ( resdf_f , resdf_s , hierarch_f_df, hierarch_s_df , soldf_f , soldf_s )


if __name__ == '__main__' :
    # from biocarta.quantification import full_mapping
    #
    adf = pd.read_csv('analytes.tsv',sep='\t',index_col=0)
    #
    adf = adf.iloc[ np.inf != np.abs( 1.0/np.std(adf.values,1) ) ,
                    np.inf != np.abs( 1.0/np.std(adf.values,0) ) ].copy()
    #
    jdf = pd.read_csv('journal.tsv',sep='\t',index_col=0)
    jdf = jdf.loc[:,adf.columns.values]
    #
    alignment_label , sample_label = 'Disease' , None
    add_labels = ['Cell-line']
    #
    cmd                = 'max'
    bVerbose           = True
    bExtreme           = True
    n_clusters         = [20,40,60,80,100]
    n_components       = None # USE ALL INFORMATION
    umap_dimension     = 2
    n_neighbors        = 20
    local_connectivity = 20.
    transform_seed     = 42
    #
    print ( adf , jdf )
    #
    #distance_type = 'correlation,spearman,absolute' # DONT USE THIS
    distance_type = 'covariation' # BECOMES CO-EXPRESSION BASED
    #
    results = full_mapping ( adf.iloc[:500,:] , jdf			,
        bVerbose = bVerbose		,
	bExtreme = bExtreme		,
	n_clusters = n_clusters		,
        n_components = n_components 	,
        bUseUmap = False 		,
        distance_type = distance_type  	,
        umap_dimension = umap_dimension	,
        umap_n_neighbors = n_neighbors	,
        umap_local_connectivity = local_connectivity ,
        umap_seed = transform_seed	,
	hierarchy_cmd = cmd		,
        add_labels = add_labels		,
	alignment_label = alignment_label ,
        bNonEuclideanBackprojection = True ,
	sample_label = None	)
    #
    map_analytes	= results[0]
    map_samples		= results[1]
    hierarchy_analytes	= results[2]
    hierarchy_samples   = results[3]

