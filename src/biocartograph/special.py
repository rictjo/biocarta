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
import numpy as np
import pandas as pd
import umap

from impetuous.quantification import single_fc_compare
from impetuous.special import unpack

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
        'PAREN_GROUP_ID TAB CHILD_GROUP_ID','or:',
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

def calculate_compositions( adf:pd.DataFrame , jdf:pd.DataFrame , label:str, bAddPies:bool=True ) -> pd.DataFrame :
    from impetuous.quantification import compositional_analysis
    from impetuous.quantification import composition_absolute
    cdf			= composition_absolute ( adf=adf , jdf=jdf , label=label )
    composition_df      = cdf.T.apply(compositional_analysis).T
    composition_df .columns = ['Beta','Tau','Gini','Geni','TSI','FILLING']
    max_quant_df        = cdf.T.apply(lambda x: x.index.values[np.argmax(x)] )
    composition_df .loc[ max_quant_df.index.values , 'Leading Quantification Label' ] = max_quant_df.values
    if bAddPies :
        from impetuous.quantification import composition_piechart
        fractions_df    = composition_piechart( cdf )
        return ( pd.concat( [composition_df.T, fractions_df]).T )
    return ( composition_df )

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
    contents = os.listdir(dir_string)
    what = []
    for thing in contents :
        if 'str' in str(type(use_exclusion)):
            if 'pruned' in thing :
                continue
        if thing.split('.')[-1] in check_set :
            what.append( [dir_string+thing ,'\t' if '.tsv' in thing else ',' if '.csv' in thing else 'X' if '.xlsx' in thing else ' ' ] )
    return ( what )

def prune_dataframe( df:pd.DataFrame , indices:str , property_name:str , value_name:str ) -> pd.DataFrame :
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
    df.loc[:,hierarchy_id+'0'] = 1 # ROOT
    #
    # exit(1)
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
                                                hierarchy_level_label:str = 'HCLN' ) -> None :
    GMTS,PCL = reformat_results_to_gmtfile_pcfile (	hierarchy_file		=  hierarchy_file ,
                                                	header_str		=  header_str ,
                                                	hierarchy_id		=  hierarchy_id ,
                                                	axis			=  axis ,
							order			=  order ,
                                                	hierarchy_level_label	=  hierarchy_level_label )

    gmt_of	= open ( header_str + hierarchy_file.split('.')[0] + '_gmts.gmt' , 'w' )
    for g in GMTS :
        print ( g, file = gmt_of)
    gmt_of.close()

    pc_of	= open ( header_str + hierarchy_file.split('.')[0] + '_pcfile.txt' , 'w' )
    print ( "parent\tchild" , file = pc_of )
    for g in PCL :
        print ( g[1]+'\t'+g[2], file = pc_of)
    pc_of.close()

def rescreen (  header_str:str          = "../results/DMHMSY_Wed_Apr__5_10_02_33_2023_" ,
                full_solution:str       = "hierarch_f.tsv" ,
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
    p   = 1. + 1. * (p<0) + p * (p>=0)
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


if __name__ == '__main__':
    reformat_results_and_print_gmtfile_pcfile(header_str = '../results/DMHMSY_Fri_Mar_17_14_37_07_2023_' )
    reformat_results_and_print_gmtfile_pcfile(header_str = '../results/DMHMSY_Fri_Mar_17_16_00_47_2023_' )


