"""
Copyright 2024 RICHARD TJÖRNHAMMAR
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

from impetuous.quantification	import single_fc_compare
from impetuous.clustering	import approximate_auc
from impetuous.special		import unpack
from impetuous.convert		import read_rds , write_rds
from impetuous.quantification	import spearmanrho , pearsonrho , tjornhammarrho, correlation_core
from impetuous.special          import hc_d2r , hc_r2d , hc_assign_lin_nn , hc_assign_lin_nnn

clean_label = lambda x : x.replace(' ','_').replace('/','Or').replace('\\','').replace('-','')
unique_list         = lambda Y : [ list(li) for li in list(set( [ tuple((z for z in y)) for y in Y ] )) ]
make_hex_color      = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)
invert_color        = lambda color : make_hex_color( [  255 - int('0x'+color[1:3],0) ,
                                        255 - int('0x'+color[3:5],0) ,
                                        255 - int('0x'+color[5:] ,0) ] )
unordered_remove    = lambda Y,Z :  [ list(li) for li in list( set( [ tuple((z for z in y)) for y in Y ])\
                                                             - set( [ tuple((x for x in z)) for z in Z ]) ) ]
list_is_in_list     = lambda Y,Z :  len( set([ tuple((z for z in y)) for y in Y ]) - set([ tuple((x for x in z)) for z in Z ]) ) == 0



def calculate_volcano_df( vals_df:pd.DataFrame , levels:list[str] , what:str='Regulation' ,
                                 bLog2:bool=False , bRanked:bool=False ) -> pd.DataFrame :
    if bLog2 :
        for idx in vals_df.index :
            if not idx == what :
                w = np.min(vals_df.loc[idx].values)
                vals_df .loc[idx] = [ np.log2( 1 + ( v - w ) ) for v in vals_df.loc[idx].values ]
    volcano_dict = single_fc_compare( vals_df , what = what , levels = levels ,
                 bLogFC = False ,  # THIS SHOULD NEVER BE SET TRUE
                 bRanked = bRanked )
    clab = ', '.join( [ 'contrast' , 'mean diff' if not bLog2 else 'log2 FC' ] )
    volcano_df = pd.DataFrame( [volcano_dict['statistic'][0] , volcano_dict['p-value'][0] , volcano_dict['contrast'][0] ] ,
                     columns=volcano_dict['index'], index = ['statistic','p-value',clab] ).T
    volcano_df.index .name = volcano_dict['comparison'][0]
    volcano_df.loc[ :, '-log10 p-value' ] = -np.log10( volcano_df.loc[:,'p-value'].values )
    return ( volcano_df )


def print_gmt_pc_file_format ( ) :
    desc__ = [ '',
	'A gmt file is a tab delimited file where each row is started by a group id'      ,
        'afterwhich a description of the group constitutes the second field. Each field'  ,
        'after the first two then corresponds to an analyte id so that the format, for each line,' ,
        'can be understood in the following way:',
	'GROUPDID TAB DESCRIPTION TAB ANALYTEID TAB ...MORE ANALYTE IDS... TAB ANALYTEID',
        'or:' ,
        'PROTF001\tTechnical words can be enlightening\tAID0001\tAID1310\tAID0135',
        '',
        'A pc file is a parent child list containing only a single parent and a single child',
        'delimited by a tab on each line so that the format, for each line, can be understood ',
        'in the following way :' ,
        'PARENT_GROUP_ID TAB CHILD_GROUP_ID','or:',
        'PROTF010\tPROTF001',
        ''
    ]
    print ( '\n'.join(desc__) )

def read_rds_distance_matrix ( filename:str ) -> np.array :
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from scipy.spatial.distance import pdist,squareform
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    return ( squareform(readRDS (filename) ) )

def inplace_norm ( x:pd.DataFrame , n:int=10 , random_state:int=42 , axis:int=0 ) -> pd.DataFrame :
    from sklearn.preprocessing import quantile_transform
    x = x.T
    y = x.values[:-1]	# REMOVE GROUPING LABEL VALUES
    if axis == 0 :	# OVER ROWS
        nvals = quantile_transform (   y ,
			n_quantiles=n , random_state=random_state ,
			copy=True )
    else :		# OVER COLUMNS
        nvals = quantile_transform ( y.T ,
			n_quantiles=n , random_state=random_state ,
			copy=True )
        nvals = nvals.T
    return ( pd.DataFrame ( nvals , index=x.index.values[:-1] , columns=x.columns.values ).T )

def quantile_class_normalisation ( adf:pd.DataFrame , classes:list[str]=None ,
                 n:int=10 , random_state:int=42, axis:int=0 ) -> pd.DataFrame :
    if classes is None :
        from scipy.stats import rankdata
        if axis==0 :
            return ( adf  .apply(lambda x:(rankdata(x.values,'average')-0.5)/len(set(x.values)))   )
        else :
            return ( adf.T.apply(lambda x:(rankdata(x.values,'average')-0.5)/len(set(x.values))).T )
    else :
        adf.loc['QuantileClasses'] = classes
        return ( adf.T.groupby('QuantileClasses') .apply( lambda x : inplace_norm( x ,
                 n=n , random_state=random_state, axis=axis ) ).T )

def symmetrize_broken_symmetry ( b_distm:np.array , method = 'average' ) -> np.array :
    a_distm = None
    if method  == 'average' :
        a_distm = np.mean(np.array([b_distm,b_distm.T]),0)
    if method  == 'max' :
        a_distm = np.max (np.array([b_distm,b_distm.T]),0)
    if method  == 'min' :
        a_distm = np.min (np.array([b_distm,b_distm.T]),0)
    if method  == 'svd' :
        from impetuous.quantification   import distance_calculation
        u_ , s_ , vt_ = np.linalg.svd( b_distm )
        a_distm = distance_calculation ( u_*s_ , 'euclidean' , False , None )
    if a_distm is None :
        a_distm = np.mean(np.array([b_distm,b_distm.T]),0)
    a_distm *= ( 1-np.eye(len(b_distm))>0 )
    return ( a_distm )

def pivot_data ( mdf:pd.DataFrame , index:str ='index' , column:str = 'sample', values:str = 'value' ) -> pd.DataFrame :
    pdf = mdf.pivot( index = index , columns = [column] , values = values )
    return ( pdf )

def check_directory ( dir_string:str ,
                      use_exclusion:str = 'pruned' ) -> list :
    check_set  = {'tsv','csv','xlsx'}
    if dir_string[-1] != '/':
        dir_string += '/'
    if not ( dir_string[0] == '~' or dir_string[0] == '/' ) :
        print ( "INCLOMPLETE DIRECTORY" )
    import os
    contents	= os.listdir(dir_string)
    what	= []
    for thing in contents :
        if 'str' in str(type(use_exclusion)):
            if 'pruned' in thing :
                continue
        if thing.split('.')[-1] in check_set :
            what.append( [dir_string+thing ,'\t' if '.tsv' in thing else ',' if '.csv' in thing else 'X' if '.xlsx' in thing else ' ' ] )
    return ( what )

def prune_dataframe ( df:pd.DataFrame , indices:str , property_name:str , value_name:str ) -> pd.DataFrame :
        from biocartograph.special import pivot_data
        pdf = pivot_data( df, index = indices , column = property_name , values = value_name )
        pdf = pdf.dropna( )
        pdf = pdf.iloc[ np.inf != np.abs( 1.0/np.std(pdf.values,1) ) ,
                        np.inf != np.abs( 1.0/np.std(pdf.values,0) ) ].copy().apply(pd.to_numeric)
        pdf .loc[:,indices] = pdf.index
        mdf = pdf.melt( id_vars = [indices] )
        mdf = mdf.rename( columns = {'value':value_name} )
        mdf = mdf.sort_values(by=[indices,property_name])
        mdf .index = mdf.loc[:,indices]
        mdf = mdf.loc[:,[property_name,value_name]]
        return ( mdf )

def create_file_lookup( files:list[str] ) -> dict :
    file_lookup = dict()
    for file in files :
        with open(file[0],'r') as input :
            for line in input :
                columns = line.replace('\n','').split(file[1])
                file_lookup[file[0]] = [ c for c in columns if len(c)>0 ]
                break
    return ( file_lookup )

def reformat_results_to_gmtfile_pcfile (	header_str:str          = '../results/DMHMSY_Fri_Mar_17_14_37_07_2023_' ,
						hierarchy_file:str	= 'resdf_f.tsv' ,
						hierarchy_id:str	= 'cids.user.'  ,
						axis:int = 0, order:str = 'descending'  , bRedundant:bool=False ,
						hierarchy_level_label:str = 'HCLN' , bErrorCheck:bool = False ) -> tuple[list] :
    #
    df = pd.read_csv( header_str+hierarchy_file , sep='\t' , index_col=0 )
    if axis == 1 :
        df = df.T
    df = df.loc[:,[c for c in df.columns if hierarchy_id in c]].apply(pd.to_numeric)
    nm = np.shape(df)
    bCreatePC = nm[1]>1
    if not bCreatePC :
        print ( 'NOT ENOUGH LABELS FOR CONSTRUCTING A HIERARCHY' )
        print ( 'SINGLE LABELING SUPPLIED : ' , hierarchy_id )
        print ( 'FLAT GMT FOR CLUSTERS WILL BE CONSTRUCTED' )
        gmtdf = df.groupby(hierarchy_id).apply(lambda x:x.index.values.tolist())
        GMTS = list()
        for name,entry in zip(gmtdf.index,gmtdf.values) :
            GMTS .append(  hierarchy_level_label + '-' + 'cluster id ' + str(name) + '\t' + str(name) + '\t' + '\t'.join(entry)  )
        return ( GMTS , None )
    df.loc[:,hierarchy_id+'0'] = 1 # ROOT
    #
    cvs = sorted([ v[::-1] for v in np.max( df,0 ).items() ] )
    if cvs[0][0] >= cvs[-1][0] :
        print ( 'WARNING: UNUSABLE ORDER' )
    if len(cvs) < 2 :
        print ( 'WARNING: TO FEW LEVELS HIERARCHY LEVELS' )
    cnvs = [ c[1] for c in cvs ]
    df   = df.loc[ : , cnvs ]
    #
    # BUILD THE LIST
    pcl = list()
    I,N = 0 , len(cnvs)
    GMTS = list()
    GMTD = dict()
    for p_c , c_c in zip(cnvs[:-1],cnvs[1:]) :
        I += 1
        dft = df.loc[ :, [p_c,c_c] ]
        dft .columns = [ c.replace(hierarchy_id,hierarchy_level_label) for c in dft.columns ]
        cols  = dft.columns.values
        nvals = []
        for vals in dft.loc[ :, cols ].values :
            pcl .append( tuple( ( I , cols[0]+'-c'+str(vals[0]),cols[1]+'-c'+str(vals[1])) )  )
            nvals .append ( pcl[-1][1:] )
        dft .loc[:,cols] = nvals
        gmt_df = dft.loc[ : , [cols[1]] ].groupby( cols[1] ).apply(lambda x:'\t'.join( [ str(w) for w in x.index]))
        for idx in gmt_df .index :
            GMTS .append ( '\t'.join( [idx,'clusters belonging to ' + str( c_c ) ,gmt_df.loc[idx]] ) )
            GMTD[idx] = set(gmt_df.loc[idx].split('\t'))
    PCL = sorted ( list( set(pcl) ) )
    if bRedundant :
        return ( GMTS, PCL )

    PCD = { p:c for n,p,c in PCL }
    if bErrorCheck:
        from collections import Counter
        v1 = [ g.split('\t')[0] for g in GMTS ]
        v2 = [ p for p in  unpack([p[1:] for p in PCL])  ]
        print ( Counter(v1) )
        s1 = set(v1)
        s2 = set(v2)
        print ( s1-s2 )
        print ( s2-s1 )
        exit(1)

    redundant = set([])
    for pc in PCL :
        if pc[1] in redundant:
            continue
        if pc[1] in GMTD and pc[2] in GMTD :
            if GMTD[ pc[1] ] == GMTD[ pc[2] ]  :
                if pc[2] in PCD :
                    PCD[ pc[1] ] = PCD[ pc[2] ]
                    del PCD[ pc[2] ]
                else :
                    del PCD[ pc[1] ]
                del GMTD[ pc[2] ]
                redundant = redundant|set([pc[2]])
    GMTS = []
    relevant_set = {  v for v in unpack( [[item[0],item[1]] for item in PCD.items()] ) }
    for item in GMTD.items():
        if item[0] in relevant_set :
            GMTS.append( '\t'.join([ item[0],'TEXT', *list(item[1]) ]) )
    PCL = []
    for item in  PCD.items() :
        PCL.append([ -1, item[0] , item[1] ])
    return ( GMTS, PCL )

def reformat_results_and_print_gmtfile_pcfile ( header_str:str          = '../results/DMHMSY_Fri_Mar_17_14_37_07_2023_' ,
						hierarchy_file:str      = 'resdf_f.tsv' ,
                                                hierarchy_id:str        = 'cids.user.' ,
                                                axis:int = 0, order:str = 'descending' ,
                                                hierarchy_level_label:str = 'HCLN' ) -> list[str] :
    GMTS,PCL = reformat_results_to_gmtfile_pcfile (	hierarchy_file		=  hierarchy_file ,
                                                	header_str		=  header_str ,
                                                	hierarchy_id		=  hierarchy_id ,
                                                	axis			=  axis ,
							order			=  order ,
                                                	hierarchy_level_label	=  hierarchy_level_label )
    gmtname,pcname = None,None
    gmtname     = header_str + hierarchy_file.split('.')[0] + '_gmts.gmt'
    gmt_of	= open ( gmtname , 'w' )
    print ( 'WRITING TO:' , gmtname )
    for g in GMTS :
        print ( g, file = gmt_of)
    gmt_of.close()

    if not PCL is None :
        pcname  = header_str + hierarchy_file.split('.')[0] + '_pcfile.txt'
        print ( 'WRITING TO:' , pcname )
        pc_of	= open ( pcname , 'w' )
        print ( "parent\tchild" , file = pc_of )
        for g in PCL :
            print ( g[1]+'\t'+g[2], file = pc_of)
        pc_of.close()
    return( gmtname , pcname )

def rescreen (  header_str:str          = "../results/DMHMSY_Wed_Apr__5_10_02_33_2023_" ,
                full_solution:str       = "hierarch_f.tsv" , bVerbose:bool=False ,
                o_filename:str          = None ) -> pd.DataFrame :
    #
    filename            = header_str + full_solution
    #
    df = pd.read_csv( filename , sep='\t', index_col=0 )
    df.index = range(len(df))
    #
    from collections import Counter
    L = len( df )
    full_result_dict = dict()
    for i in range( L ) :
        arow = df.iloc[i,:].values
        for item in Counter( list( Counter( arow ).values() ) ).items() :
            if item[0] in full_result_dict :
                full_result_dict[ item[0] ] += item[1]
            else :
                full_result_dict[ item[0] ]  = item[1]
        if bVerbose:
            print ( 'DONE' , i , 'OF' , L, 'OR',i/L )
    df_ = pd.DataFrame( full_result_dict.items() , columns=['k','v'] )
    if not o_filename is None :
        df_.to_csv( o_filename , sep='\t' )
    return ( df_ )

def contraction ( value:float , q:float , d:float ) -> float :
    if value <= 0 :
        return ( 0.0 )
    r   = value / q # wasteful slob ...
    r2  = r*r
    r4  = r2*r2
    r6  = r4*r2
    r12 = r6*r6
    p   = ( 1/r12 - 1/r6 ) * d * (-1)
    p   = 1. + 0. * (p<0) + p * (p>=0)
    return ( value / p ) # WARNING ONLY FOR WASTEFUL SAIGAS

def contract ( a:np.array , d:float=None , q:float=None , quantile:float=0.05 ) -> np.array :
    nm = np.shape(a)
    b  = a.reshape(-1)
    d_,q_ = d,q
    if d is None :
        d_ = 1.0/list(set(sorted( b )))[1]
    if q is None :
        q_ = np.quantile( b , q=quantile )
    # THIS OPERATION IS SLOW
    P  = np.array(list(map( lambda x:contraction(x , q=q_ , d=d_) , b ))).reshape(nm)
    return ( P )

def contract_df ( a:pd.DataFrame , d:float=None , q:float=None , quantile:float=0.05 ) -> pd.DataFrame :
    nm = np.shape(a.values)
    b  = a.values.reshape(-1)
    d_,q_ = d,q
    if d is None :
        d_ = 1.0/list(set(sorted( b )))[1]
    if q is None :
        q_ = np.quantile( b , q=quantile )
    # THIS OPERATION IS SLOWER
    return ( a.apply( lambda x : pd.Series([contraction(x_ , q=q_ , d=d_ ) for x_ in x],name=x.name,index=x.index) )  )

def generate_hulls ( df:pd.DataFrame , gid:str = 'cids.max' , hid:str='UMAP.' ,
                     cid:str = None , xid:str = None ,
                     incremental:bool=False, qhull_options:str = None , bPlottered:bool=False ) -> dict :
    #
    # CONSIDERING SWTICH FROM QHULL TO FASTER AND TOPOLOGICALLY
    # BETTER ALTERNATIVE IN THE FUTURE
    #
    if cid is None :
        cid = hid
    selection = sorted( list( set([ c for c in df.columns if hid in c or gid == c ]) - set([gid]) ) )
    selection .append(gid)
    projection_crds = sorted( list( set([ c for c in df.columns if cid in c ])  ) )
    if not xid is None :
        selection = [ s for s in selection if not xid in s ]
        projection_crds = [ s for s in projection_crds if not xid in s ]

    df_used = df.loc[ :,selection ]
    nm = np.shape(df_used.values)
    if nm[-1] < 3 :
        print ( 'WARNING: MUST HAVE AT LEAST 2D DATA AND A LABEL SPECIFIED' )
        print ( '         IN A DATAFRAME SO THAT DIM(DF) = N,M WHERE M>=2 ' )
        print ( '         WILL RETURN EMPTY SOLUTION' )
    import scipy.spatial as scs
    if bPlottered :
        import matplotlib.pyplot as plt
    hulled = df_used.groupby(gid).apply( lambda x: tuple(( df.loc[x.index,projection_crds].values.tolist() ,
                         scs.ConvexHull(        [ v[:-1] for v in x.values.tolist() ] ,
                                                incremental=incremental, qhull_options=qhull_options ) )) )
    all_convex_hulls = {}
    for J in range(len( hulled )) :
        gid_nr		= hulled.index.values[J]
        ahull		= hulled.iloc[J]
        points		= np.array( ahull[0] )
        convex_hull	= ahull[1]
        if bPlottered :
            plt .plot ( points[:,0],points[:,1],'o' )
        hull_border  = [ [ *points[convex_hull.vertices,i], points[convex_hull.vertices[0],i] ] for i in range(len(projection_crds)) ]
        convex_hull_center  = np.mean(points,0)
        all_convex_hulls[ gid_nr ] = { 'area':convex_hull.area , 'center':convex_hull_center , 'border':hull_border,
					    'info' : 'scipy spatial ConvexHull : ' + qhull_options if not qhull_options is None else 'default settings' }
        if bPlottered :
            plt.plot ( hull_border[0] , hull_border[1] , 'r-')
            plt.plot ( convex_hull_center[0] , convex_hull_center[1] , '*k' )
    if bPlottered :
        plt .show()
    return ( all_convex_hulls )

def traverse_hierarchical_dictionary ( di:dict ) :
    item = di
    if not item is None :
        for k_ in item.keys() :
            print ( k_ )
            traverse_hierarchical_dictionary ( item[k_] )

def generate_neighbor_distance_df ( df:pd.DataFrame , lab:str='PCA' , iex:int=-1 , nNN:int = 20 ) -> pd.DataFrame :
    from scipy.stats import rankdata
    df0 = df.loc[ : ,[c for c in df.columns if lab in c ] ]
    if True :
        output = []
        spr = spearmanrho( df0.iloc[:,:iex],df0.iloc[:,:iex] )
        dsp = 1 - spr
        for i in range(len( df0 )) :
            j    = df0.index.values
            m    = j[i]
            x    = dsp[i]
            bSel = rankdata ( x , 'ordinal' ) - 1 < nNN
            output = [ *output , *sorted( [tuple((d_,r_,m,n_)) for d_,r_,n_ in  zip(  x[bSel], spr[i][bSel] ,j[bSel] )] ) ]
        return ( pd.DataFrame(output,columns=['cor_distance','cor','gene','neighbor_gene']).loc[:,['gene','neighbor_gene','cor_distance','cor']] )


def generate_neighbor_distance_information(  df:pd.DataFrame , lab:str='PCA' , iex:int=-1 , nNN:int = 20 ,sep:str='\t' ) -> str :
    # results/Clustering_results/brain_HPA23v4/distance/nearest_neighbors.tsv
    nn_df = generate_neighbor_distance_df ( df=df , lab=lab , iex=iex , nNN=nNN )
    file_info = sep.join([ str(c) for c in nn_df.columns.values.tolist() ]) + '\n'
    for v in nn_df.values :
        file_info += sep.join( [ str(w) for w in v ] ) + '\n'
    return ( file_info )


def cluster_center_information ( all_convex_hulls:dict , sep:str = '\t' ) -> str :
    # results/Clustering_results/brain_HPA23v4/UMAP/cluster_centers.tsv
    N           = len( list(all_convex_hulls.items())[0][1]['center'] )
    crdn        = 'xyzuvwabcdefghijklmnopqrst'
    file_info   = ""
    if len( crdn ) < N :
        print ( 'WARNING : THERE IS PROBABLY AN ERROR IN THE HULL CALCULATION ', N , len(crdn) )
    #
    file_info += sep.join( [ 'cluster' , *[ crdn[i] for i in range(N) ] ] ) + '\n'
    for item in all_convex_hulls.items() :
        file_info += str(item[0]) + sep + sep.join([str(v) for v in item[1]['center']] ) + '\n'
    return ( file_info )


def create_cluster_polygon_information ( all_convex_hulls:dict , sep:str = '\t' ) -> str :
    # results/Clustering_results/brain_HPA23v4/UMAP/UMAP_polygons.tsv
    DIM         = len( list(all_convex_hulls.items())[0][1]['center'] )
    crdn        = 'xyzuvwabcdefghijklmnopqrst'.upper()
    file_info   = ""
    if len( crdn ) < DIM :
        print ( 'WARNING : THERE IS PROBABLY AN ERROR IN THE HULL CALCULATION ', DIM , len(crdn) )
    #
    file_info += sep.join( [    'cluster' , 'sub_cluster' , 'landmass', 'sub_type' ,
                                *[ crdn[i] for i in range(DIM) ] ,
                                'L1' , 'L2' , 'polygon_id'] ) + '\n'
    #
    for item in all_convex_hulls.items() :
        crds = item[1]['border']
        M    = len(crds[0])
        crds = np.array(crds).reshape(-1)
        crds = [ [ crds[k*M+i] for k in range(DIM) ] for i in range(M) ]
        for crd in crds :
            file_info += sep.join([ str(int(float(item[0]))) , '1' , '1' ,'primary' , *[str(c) for c in crd] , '1' , '1' ,   str(int(float(item[0])))+'_1_1' ]) + '\n'
    return ( file_info )


def create_directory ( directory_path:str ) :
    import os
    drsp = directory_path.split('/')
    for i in range( len(drsp) ) :
        if drsp[i] == '.' or drsp[i] == '..' :
            continue
        if len(drsp[:i+1])>0 :
            thisdir = '/'.join(drsp[:i+1])
            l = set( os.listdir ( '/'.join(thisdir.split('/')[:-1]) ) )
            if not thisdir.split('/')[-1] in l :
                try:
                    os.mkdir( thisdir )
                except:
                    print ( "WOULD NOT CREATE DIRECTORY:",thisdir,"\n\t\t(WILL PRETEND IT EXISTS ANYWAY)")

#
def quick_check_solution(header_str:str = '../results/DMHMSY_Tue_May__2_10_57_05_2023_' ) :
    file 	= header_str + 'soldf_f.tsv'
    df		= pd.read_csv(file,sep='\t',index_col=0 )
    print ( df.iloc[:,np.argmax(df.iloc[1,:].values)]   )


def simple_membership_inference( header_str:str ) :
        df = pd.read_csv(  header_str[:-1] + '/clustering/final_consensus.tsv' ,sep='\t' ,index_col = 0 )
        cs = sorted(list(set( df.loc[:,'cluster'].values.tolist() )))
        N  = len(cs)
        n_trials = 100.
        uncertainty = 0.5
        dfmc = np.abs(pd.DataFrame( [ cs==v for v in df.values ] , index=df.index )*N*n_trials+uncertainty )
        dfmc .columns = cs
        dfmc = ( dfmc.T / np.sum(dfmc,1) ) .T
        #
        df_  = pd.DataFrame( [ q for q in zip( dfmc.values.reshape(-1) ,
                [ v for w in dfmc.index.values for v in dfmc.columns.values ] ,
                [ w for w in dfmc.index.values for v in dfmc.columns.values ] ) ] )
        df_ .index      = df_.iloc[:,-1]
        df_ .index.name = 'gene'
        df_             = df_.iloc[:,[0,1]]
        df_ .columns    = ['membership','cluster']
        df_ .loc[:,'nclust'] = [ int(c) for c in df_.loc[:,'cluster'].values.tolist() ]
        df_ = df_.sort_values('nclust').loc[:,['membership','cluster']]
        df_ .to_csv( header_str[:-1] + '/clustering/cluster_memberships.tsv',sep='\t')

def coexpression_distance_matrix (	coordinates_file:str , coordinate_label:str = 'PCA' ,
                                        pathfile:str = None ,
					path:str = './' , filename:str = 'distances.tsv.gz' , sep='\t' ) :
    if pathfile is None :
        pathfile = path+filename
    try :
        Ddf	= pd.read_csv( pathfile , sep = sep , compression = 'infer' , index_col = 0 )
        D	= Ddf.values
    except :
        print ( "WRITING A COVARIATION BASED DISTANCE MATRIX" )
        cdf         = pd.read_csv( coordinates_file , sep=sep , index_col=0 )
        cdf         = cdf.iloc[:,[ coordinate_label in c for c in cdf.columns ] ]
        from scipy.spatial.distance import pdist , squareform
        D = squareform ( pdist( cdf.values ) )
        Ddf = pd.DataFrame ( D , columns = cdf.index.values ,
                        index = cdf.index.values )
        Ddf .to_csv( pathfile , sep = sep , compression = 'infer' )
    return ( D )
#
def append_comprehensive_cluster_information ( header_str:str ,
                        cluster_file:str                = "clustering/final_consensus.tsv" ,
                        neighbor_file:str               = "distance/nearest_neighbors.tsv" ,
                        distance_matrix:np.array        = None , fraction:float=0.1 ,
                        bVerbose:bool                   = True ) :

    from impetuous.clustering	import complete_happiness , complete_immersiveness
    from scipy.spatial.distance	import squareform

    ldf         = pd.read_csv( header_str + cluster_file  , sep='\t' , index_col=0)
    ndf         = pd.read_csv( header_str + neighbor_file , sep='\t' )
    cost        = header_str + "clustering/cluster_comprehensiveness.tsv"
    #
    happ = complete_happiness( ldf , ndf )
    if bVerbose :
        print ( happ )
    ldf .loc[ : ,    'happiness' ] = happ
    ldf .to_csv( cost , sep='\t' )
    if distance_matrix is None :
        return
    nm = np.shape( distance_matrix )
    if nm[0] != len(ldf) :
        print ( "ERROR COULD NOT CALCULATE IMMERSIVE STATS DUE TO MALFORMED DISTANCE MATRIX" )
        exit(1)
    ci_ = complete_immersiveness ( ldf.iloc[:,0].values.tolist() , distance_matrix , fraction=fraction )
    if bVerbose :
        print ( ci_ )
    ldf .loc[ : , 'immersiveness' ]     = [ c[0] for c in ci_ ]
    ldf .loc[ : , 'SE(immersiveness)']  = [ c[1] for c in ci_ ]
    ldf .to_csv( cost , sep='\t' )


def generate_atlas_files ( header_str:str ,
                fcfile:str = 'clustering/final_consensus.tsv' ,
		nnfile:str = 'distance/nearest_neighbors.tsv' ,
                umfile:str = 'UMAP/UMAP.tsv' ,
                ccfile:str = 'UMAP/cluster_centers.tsv' ,
		pofile:str = 'UMAP/UMAP_polygons.tsv' ,
                anfile:str = 'enrichment/cluster_annotations.tsv',
                difile:str = 'distance/distances.tsv.gz' ,
                pcadir:str = 'PCA/',
                evadir:str = 'evaluation/',
                enrichment_results_file_pattern:list[str] = ["(HEADER)treemap_c(CLUSTID).tsv"] ,
                nNN:int=20 , bConcise:bool = False , fraction:float=0.1 ,
                additional_directories:list[str] = ['data','enrichment','evaluation','graph','PCA','UMAP',
					'svg'  , 'svg/heatmap','svg/bubble','svg/treemap','svg/fountain',
                                        'html' , 'html']) :
    #
    # START ATLAS GEN
    sep = '\t'
    #
    hdir_str = header_str
    if hdir_str[-1] == '_' :
        hdir_str =  hdir_str[:-1] + '/'
    elif  hdir_str[-1] != '/' :
        hdir_str += '/'
    #
    df_sol_     = pd.read_csv( header_str + 'resdf_f.tsv' , sep=sep , index_col=0 ) # SHOULD BE ASSERTED
    df_pca_     = pd.read_csv( header_str + 'pcas_df.tsv' , sep=sep , index_col=0 ) # SHOULD BE ASSERTED
    df_pca_s    = pd.read_csv( header_str + 'pcaw_df.tsv' , sep=sep , index_col=0 ) # SHOULD BE ASSERTED
    if 'str' in str(type(pcadir)) :
        if pcadir[-1] != '/' :
            pcadir += '/'
        create_directory ( hdir_str + pcadir )
        df_pca_ .to_csv( hdir_str + pcadir + 'pca_analytes_df.tsv' )
        df_pca_s.to_csv( hdir_str + pcadir + 'pca_samples_df.tsv'  )
    #
    if 'str' in str(type(evadir)) :
        if evadir[-1] != '/' :
            evadir += '/'
        create_directory ( hdir_str + evadir )
        e1df = pd.read_csv( header_str + 'soldf_f.tsv' , sep=sep , index_col=0 )
        e1df .to_csv( hdir_str + evadir + 'solution_analytes.tsv' ,sep=sep )
        del e1df
        e2df = pd.read_csv( header_str + 'soldf_s.tsv' , sep=sep , index_col=0 )
        e2df .to_csv( hdir_str + evadir + 'solution_samples.tsv' , sep=sep )
        del e2df
        c1df = pd.read_csv( header_str + 'composition.tsv'  , sep=sep , index_col=0 )
        c1df .to_csv( hdir_str + evadir + 'composition.tsv' , sep=sep )
        del c1df
    #
    common_idx  = sorted( list( set( df_sol_.index.values ) & set( df_pca_.index.values )))
    df_sol_     = df_sol_.loc[ common_idx,: ]
    df_pca_     = df_pca_.loc[ common_idx,: ]
    #
    df		= df_sol_
    minmax      = lambda x: np.array( [ np.min(x,0) , np.max(x,0) ] )
    scale       = minmax ( df.loc[:,[c for c in df if 'UMAP.' in c ]].values )
    dfs         = ( df .loc[:,[c for c in df if 'UMAP.' in c ]] - scale[0]) / (scale[1]-scale[0])
    dfs.columns = [ str(c) + '.scaled' for c in dfs.columns ]
    df          = pd.concat([df.T,dfs.T]).T
    all_hulls   = generate_hulls( df , hid = '.scaled' ,
                        bPlottered=False )
    #
    create_directory ( hdir_str + '/'.join( fcfile.split('/')[:-1]) )
    final_consensus_df = df_sol_.loc[:,['cids.max']].rename(columns={'cids.max':'cluster'}).copy()
    final_consensus_df .index.name = 'gene'
    final_consensus_df .to_csv( hdir_str + fcfile, sep=sep )
    #
    create_directory ( hdir_str + '/'.join( umfile.split('/')[:-1]) )
    udf = df.loc[:,[c for c in df.columns if 'UMAP' in c ] ]
    udf .columns = [ c.replace('.','_') for c in udf.columns ]
    udf = udf .rename( columns={'UMAP_0_scaled':'UMAP_1_scaled','UMAP_1_scaled':'UMAP_2_scaled'} )
    udf .to_csv( hdir_str + umfile, sep=sep )
    #
    create_directory ( hdir_str + '/'.join( ccfile.split('/')[:-1]) )
    ccinfo = cluster_center_information( all_hulls )
    o_f = open( hdir_str + ccfile,'w')
    print ( ccinfo , file=o_f )
    o_f .close()
    #
    create_directory ( hdir_str + '/'.join( pofile.split('/')[:-1]) )
    poinfo  = create_cluster_polygon_information( all_hulls )
    o_f = open( hdir_str + pofile,'w')
    print ( poinfo , file=o_f )
    o_f .close()
    #
    create_directory ( hdir_str + '/'.join( nnfile.split('/')[:-1]) )
    nninfo  = generate_neighbor_distance_information(  df_pca_ , lab = 'PCA' ,
			 iex = -1 , nNN = nNN , sep = '\t' )
    o_f = open( hdir_str + nnfile,'w')
    print ( nninfo , file=o_f )
    o_f .close()
    for dir in additional_directories :
        create_directory ( hdir_str + dir )
    #
    # CHECK FOR ENRICHMENTS
    hdr_dir   = '/'.join( header_str.split('/')[:-1] )
    id_tag    = header_str.split('/')[-1]
    import os
    enr_files = [ l for l in set(os.listdir(hdr_dir)) if id_tag in l and ('treemap' in l or 'GFA' in l or 'enrichment' in l ) ]
    if len( enr_files ) > 0 :
        for f in enr_files :
            os.system('cp ' + hdr_dir + '/' + f + ' ' +  hdir_str + 'enrichment/' + f )
    from biocartograph.enrichment import auto_annotate_clusters
    an_df	= auto_annotate_clusters ( header_str , enrichment_results_file_pattern = enrichment_results_file_pattern )
    create_directory ( hdir_str + '/'.join( anfile.split('/')[:-1]) )
    an_df	.to_csv( hdir_str + anfile , sep='\t' )
    #
    simple_membership_inference( header_str )
    #
    if not bConcise :
        print ( "DONE : NOW WE CREATE SOME ADDITIONAL BENCHMARKS FOR THE ANALYTES" )
        print ( "YOU CAN SKIP THIS BY SETTING bConcise=True IN THE generate_atlas_files ROUTINE" )
        D = coexpression_distance_matrix ( header_str + 'pcas_df.tsv' , coordinate_label = 'PCA' ,
                                           pathfile = hdir_str + difile , sep='\t' )
        print ( "WROTE SUPPORTING COEXPRESSION DISTANCE MATRIX" )
        #
        append_comprehensive_cluster_information ( hdir_str ,
                        cluster_file			= fcfile ,
                        neighbor_file			= nnfile , fraction=fraction ,
                        distance_matrix			= D , bVerbose=False )
        print ( "PRODUCED AUC AND HAPPINESS FOR ALL!" )


from impetuous.convert import NodeGraph, Node

class DrawGraphText ( object ) :
    def __init__ ( self ,	color_label:str		= None ,
				area_label:str		= None ,
				celltext_label:str	= None ,
				font:str		= 'Arial' ,
				header:str		= None ) :
        from impetuous.quantification import grouper
        self.grouper = grouper
        #
        self.id_        :str	= ""
        self.label_     :str	= ""
        if header is None :
            self.header_        :str	= """graph {\nlayout=patchwork\nnode [style=filled]\n"""
        else :
            self.header_        :str	= header
        self.story_             :str	= self.header_
        self.base_size_         :int	= 30
        self.default_color_	:str	= "#AFFFE1"
        self.color_label_	:str	= ".#-"
        self.area_label_	:str	= "-#."
        self.celltext_label_	:str	= "#-."
        if not color_label	is None :
            self.color_label_		= color_label
        if not area_label	is None :
            self.area_label_		= area_label
        if not celltext_label	is None :
            self.celltext_label_	= celltext_label
        if not font   is None :
            self.celltext_font_        = font
        self.dt_check	= lambda data , strtype , alternative : data if strtype in str(type(data)).lower() else alternative
        self.ddt_check	= lambda data , label , strtype , alternative : self.dt_check(data[label],strtype,alternative) if label in data else alternative
        self.regroup	= lambda label , NG : '\n'.join( [ ' '.join( g ) for g in self.grouper( label.split(' '),NG) ])
        self.contract_cell_text = lambda txt,L : txt if False else self.regroup( txt , 2 )

    def create_gv_node_info ( self , node_id:str , nG:NodeGraph , bIsChild:bool=False ) :
        base_size = self.base_size_
        graphnode = nG .get_node( node_id )
        children = graphnode.get_links('descendants')
        if len( children ) == 0 :
            celltext = self.ddt_check(graphnode.get_data() , self.celltext_label_ , "str" , graphnode.identification())
            cellsize = int(np.round( self.ddt_check( graphnode.get_data() , self.area_label_ , "float" , self.base_size_  )))
            celltext = self.contract_cell_text(celltext,cellsize)
            # print ( celltext )
            nochildren = """$LABEL$  [area=$#$ fontname=$FONT$ fillcolor="$COLOR$" ]\n"""\
                .replace( '$FONT$'	, self.celltext_font_ )\
                .replace( "$LABEL$"	, '\"' + celltext + '\"' )\
                .replace(  "$#$"	, str (  cellsize  ) )\
                .replace( "$COLOR$"	, self.ddt_check( graphnode.get_data() , self.color_label_ , "str" , self.default_color_ ) )
            self.story_ += nochildren
        else :
            subheader = """subgraph \"$LABEL$\" {\nlayout=patchwork\nstyle=filled\n""".replace("$LABEL$",node_id )
            self.story_ += subheader
            for child in children :
                self.create_gv_node_info( child , nG , True )
            self.story_ += "}\n"

    def print_story ( self ) :
        print ( self.story_	+ '\n}' )

    def return_story ( self ) -> str :
        return ( self.story_	+ '\n}' )
#
# END OF CLASS
#
def create_NodeGraph_object_from_treemap_file( treemap_filename:str = '../bioc_results/DMHMSY_Fri_Feb__2_13_16_01_2024_treemap_c4.tsv',
		bHelped:bool=False , mappings:list = None , transforms:list = None ) -> NodeGraph :
    #
    make_hex_colors = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)
    dft = pd.read_csv( treemap_filename , sep='\t' , index_col = 0 )
    nG = NodeGraph()
    #
    all_names,all_added	= [] , []
    pcs			= []
    sig_maxval	, sig_minval	= np.max( -1.0*np.log10(dft.loc[:,'p-value'].values) ) , np.min(  -1.0*np.log10(dft.loc[:,'p-value'].values)  )
    num_maxval	, num_minval	= np.max( dft.loc[:,'N_intersect']  ) , np.min( dft.loc[:,'N_intersect'] )
    #
    for i in range( len(dft) ) :
        nodeid	= dft.iloc[i].loc['name']
        label	= ' : '.join( dft.iloc[i].loc[['name','description']].values.tolist() )
        v_ids	= [ dft.iloc[i].loc['parent'] ]
        #
        all_names   .append( v_ids[0] )
        all_names   .append(  nodeid  )
        all_added   .append(  nodeid  )
        #
        n_node = Node()
        n_node .set_id( nodeid )
        n_node .add_label( nodeid )
        n_node .add_description( label )
        n_node .get_data()['Description']	= label.replace( ' : ' , '\n ' ).replace('/',' / ')
        n_node .get_data()[ 'Area' ]		= 40 * dft.iloc[i].loc['N_intersect']/( num_maxval-num_minval ) + 30
        n_node .get_data()['Significance']	= -1.0* np.log10( dft.iloc[i].loc['p-value'] )
        #
        blaze = n_node .get_data()['Significance']/(sig_maxval-sig_minval) * 512
        r = int( np.ceil( 255 if blaze>=255 else blaze ))
        gb= int( np.ceil(  0  if blaze <255 else blaze-255 ))
        n_node .get_data()['Color' ]                = make_hex_colors( [ r , gb , gb] )
        #
        n_node .add_links( v_ids  , bClear=True , linktype = 'ascendants' )
        nG.add( n_node )

        pc = [ v_ids[0] , nodeid ]
        pcs.append( pc )

    if len( set(all_names)-set(all_added) )>0 :
        nodeid      = list(  set(all_names)-set(all_added) )[0]
        label       = nodeid + ' : ROOT'
        v_ids       = [ "" , "" ]
        n_node = Node()
        n_node .set_id( nodeid )
        n_node .add_label( nodeid )
        n_node .add_description( label )
        nG.add( n_node )
        nG.set_root_id(nodeid)

    for pc in pcs :
        nG.get_node(pc[0]).add_links([pc[1]],linktype='descendants')

    if bHelped :
        print ( """
    n_node.get_data()['Description']
    n_node.get_data()['Color']
    n_node.get_data()['N']
    n_node.get_data()['Significance']""" )
    return ( nG )

unique_list         = lambda Y : [ list(li) for li in list(set( [ tuple((z for z in y)) for y in Y ] )) ]
make_hex_color      = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)
invert_color        = lambda color : make_hex_color( [  255 - int('0x'+color[1:3],0) ,
                                        255 - int('0x'+color[3:5],0) ,
                                        255 - int('0x'+color[5:] ,0) ] )
#
#
def create_color ( num:int ) :
    # THE COMMON PERCEPTION OF THE VISIBLE SPECTRUM
    make_hex_color = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)
    c	= [ 0 , 0 , 0 ]
    cas	= 1 + int ( num / 256 )
    vol	= num	%	256
    if   cas == 1 :
        c[0] = vol
    elif cas == 2 :
        c[0] = 255
        c[1] = vol
    elif cas == 3 :
        c[0] = 255 - vol
        c[1] = 255
    elif cas == 4 :
        c[0] = 0
        c[1] = 255
        c[2] = vol
    elif cas == 5 :
        c[0] = 0
        c[1] = 255 - vol
        c[2] = 255
    elif cas == 6 :
        c[0] = vol
        c[1] = 0
        c[2] = 255
    elif cas == 7 :
        c[0] = 255
        c[1] = vol
        c[2] = 255
    else :
        c = [ 255 , 255 , 255 ]
    return ( make_hex_color( c ) )


def create_hilbertmap ( nG:NodeGraph		,
        quant_label:str = 'Area'	        , # quant_label = 'Significance'
        search_type:str = 'breadth'      	, # search_type = 'depth'
	n:int = 32 				) -> dict :
        m = n
        from impetuous.special import hc_d2r , hc_assign_lin_nn
        #
        tot             =   0
        extends_to      = n * m
        fractions       = dict()
        #
        for p in nG.retrieve_leaves( nG.get_root_id() , search_type )['path'] :
            tot += nG .get_graph()[p].get_data()[quant_label]
            fractions [ p ] = nG.get_graph()[p].get_data()[quant_label]
        things  = [ [ k , int(np.round(v*n*n/tot)) ] for k,v in fractions.items() ]
        correction      = n * m - np.sum([ t[1] for t in things ])
        c_s             = np.sign( correction )
        #
        for j in [ int(i) for i in np.floor(np.random.rand( np.abs(correction) ) * len(things)) ] :
            things[j][1] += c_s
        #
        # NOW THE SAME FOR COLORS
        s       =   0
        q       = 1792  # max color
        colored_things = []
        for thing in things :
            pos  = 0.5 * thing[ 1 ] + s
            s   += thing[1]
            color        = create_color( int( np.round( pos ) * q / extends_to ) )
            colored_things.append( [ thing[0] , color , pos , thing[1] ] )
        #
        dR      = dict()
        R , P , NN       = [] , [] , []
        I       = 0
        d , s_  = 0 , 0
        while ( d < extends_to ) :
            rt = hc_d2r( n , d )
            NN .append(  hc_assign_lin_nn( n , d )[1:] )
            R  .append( rt )
            d  += 1
            s_ += 1
            thing = colored_things[ I ]
            P .append([ d , *rt , thing[0] , thing[1] ] )
            if thing[0] in dR :
                dR[thing[0]] .append( rt )
            else :
                dR[thing[0]] = [ thing[1] ]
                dR[thing[0]] .append( rt )
            if s_ == thing[3] :
                s_ = 0
                I += 1
        dR['P data']	= P
        dR['NearestN']	= NN
        return ( dR )
#
def solve_border_salesman( grid_points:list , bBrute:bool=False ) -> np.array :
    #
    # SOLVES A TRAVELING SALESMAN PROBLEM WHEN THE POOR SAIGASELLER IS
    # FORCED TO MOVE ALONG BORDER POINTS
    # CAN BE USED TO DRAW A NON-CONVEX POLYGON
    #
    # CONTAINS UNRESOLVED "FEATURE" WHEN DEALING WITH SINGLE LINKED BORDER POINTS
    #
    if True :
        if True :
            n = np.max(grid_points) + 1
            val_grid_points     = grid_points.copy()
            if bBrute :
                val_grid_points = [ val_grid_points[0] ]
            solution = None
            FULL_PATH_LENGTH	= n*n
            #
            for p_entry in val_grid_points :
                sorted_coords       = [ p_entry ]
                visited_coords      = sorted_coords
                unsorted_coords     = unordered_remove( grid_points , [ p_entry ] )
                #
                bClosed = False
                x0 = p_entry
                xp = x0
                #
                while len(unsorted_coords) > 0 :
                    L  = (n + 1)*n
                    Y  = np.array( unsorted_coords     )
                    y0 = np.array(   sorted_coords[-1] )
                    ya = None
                    yL = []
                    for y_ in Y :
                        y = y_[:2]
                        J = np.sqrt( np.sum((y0-y)**2) )
                        if J <= L :
                            L  = J
                            A  = np.arctan2( *((y_- y0)[::-1]) )
                            ya = y_
                            yL .append( [J,A,*y_[:2]] )
                    if len(yL) > 0 :
                        ya = sorted([ y[1:] for y in yL if y[0] == L ])[::-1][0][1:]
                    if ya is None :
                        print ( "WARNING : EXPECTED FEATURE ENCOUNTERED !!!" )
                        print ( 'GP>' , grid_points ,'\nY>',Y,'\nL>',L,'\nyL>',yL )
                    sorted_coords   .append( ya[:2] )
                    unsorted_coords = unordered_remove( Y , [ya] )
                # SHORTEST PATH DETERMINED
                #
                sorted_coords.append(p_entry) # THE LAST POINT IS THE STARTING POINT
                sorted_coords = np.array( sorted_coords )
                PL = 0
                for i in range(1,len(sorted_coords)) :
                    PL += np.sqrt( np.sum( (sorted_coords[i-1]-sorted_coords[i])**2 ) )
                if PL < FULL_PATH_LENGTH : # MAKE IT INDEPENDENT OF STARTING POINT
                    solution = sorted_coords
                    FULL_PATH_LENGTH = PL
            return ( solution )


def solve_traveling_salesman( points:np.array , bBrute:bool=False ) -> np.array :
    #
    # SOLVES A TRAVELING SALESMAN PROBLEM WHEN THE POOR SAIGASELLER IS
    # FORCED TO MOVE ALONG BORDER POINTS
    # CAN BE USED TO DRAW A NON-CONVEX POLYGON
    #
    # THIS IS THE OFFGRID VERSION
    #
    # CONTAINS UNRESOLVED "FEATURE" WHEN DEALING WITH SINGLE LINKED BORDER POINTS
    #
    #from scipy.spatial.distance import pdist,squareform
    #LM = np.sum( np.max(squareform(pdist( points ,'euclidean')),0) )*1.01
    LM = np.inf
    if True :
        if True :
            n = np.max(points) + 1
            val_points     = points.copy()
            if bBrute :
                val_points = [ val_points[0] ]
            solution = None
            FULL_PATH_LENGTH    = LM
            #
            for p_entry in val_points :
                sorted_coords       = [ p_entry ]
                visited_coords      = sorted_coords
                unsorted_coords     = unordered_remove( points , [ p_entry ] )
                #
                bClosed = False
                x0 = p_entry
                xp = x0
                #
                while len(unsorted_coords) > 0 :
                    L  = LM
                    Y  = np.array( unsorted_coords     )
                    y0 = np.array(   sorted_coords[-1] )
                    ya = None
                    yL = []
                    for y_ in Y :
                        y = y_[:2]
                        J = np.sqrt( np.sum((y0-y)**2) )
                        if J <= L :
                            L  = J
                            A  = np.arctan2( *((y_- y0)[::-1]) )
                            ya = y_
                            yL .append( [J,A,*y_[:2]] )
                    if len(yL) > 0 :
                        ya = sorted([ y[1:] for y in yL if y[0] == L ])[::-1][0][1:]
                    if ya is None :
                        print ( "WARNING : EXPECTED FEATURE ENCOUNTERED !!!" )
                        print ( 'GP>' , points ,'\nY>',Y,'\nL>',L,'\nyL>',yL )
                    sorted_coords   .append( ya[:2] )
                    unsorted_coords = unordered_remove( Y , [ya] )
                # SHORTEST PATH DETERMINED
                #
                sorted_coords.append(p_entry) # THE LAST POINT IS THE STARTING POINT
                sorted_coords = np.array( sorted_coords )
                PL = 0
                for i in range(1,len(sorted_coords)) :
                    PL += np.sqrt( np.sum( (sorted_coords[i-1]-sorted_coords[i])**2 ) )
                if PL < FULL_PATH_LENGTH : # MAKE IT INDEPENDENT OF STARTING POINT
                    solution = sorted_coords
                    FULL_PATH_LENGTH = PL
            return ( solution )


if __name__ == '__main__' :
    #
    nG_ = create_NodeGraph_object_from_treemap_file( '../bioc_results/DMHMSY_Fri_Feb__2_13_16_01_2024_treemap_c4.tsv' )
    #
    if True :
        print ( "THE JSON DATA" )
        print ( nG_.write_json() )
        print ( "THE LEAF NODES" )
        print ( nG_.retrieve_leaves( nG_.get_root_id() ) )
    #
    dgt = DrawGraphText(	color_label = 'Color' , area_label = 'Area',
			celltext_label = 'Description' , font = 'Arial' )
    dgt .create_gv_node_info( nG_.get_root_id() , nG_  )
    graphtext = dgt.return_story()
    #
    import pygraphviz as pgv
    G1 = pgv.AGraph( graphtext )
    G1 .layout()
    G1 .draw("file1.svg")

    #
    reformat_results_and_print_gmtfile_pcfile( header_str = '../results/DMHMSY_Fri_Mar_17_14_37_07_2023_' )
    reformat_results_and_print_gmtfile_pcfile( header_str = '../results/DMHMSY_Fri_Mar_17_16_00_47_2023_' )
    #
    results_dir = '../results/'
    header_str  = results_dir + 'DMHMSY_Wed_May__3_11_48_58_2023_'
    #
    generate_atlas_files ( header_str = header_str )
