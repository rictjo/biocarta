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
import numpy  as np
import os

def create_specificity_group_df ( acdf:pd.DataFrame , df_:pd.DataFrame , index_order:list=None ,
                                  label:str = '' , post_fix:str = '' , bSorted:bool=True , slab:str = 'Specificity Class' ,
                                  rename_cats:dict = { '0':'ND' , '1':'LS' , '2':'GE' , '3':'GR' , '4':'TR' } ) :
        from impetuous.quantification import group_classifications , composition_absolute, compositional_analysis
        from collections import Counter
        lab1		= label
        results		= group_classifications ( acdf.apply(pd.to_numeric) , bLog2=True )
        new		= [ {w:item[0] for w in item[1]} for item in results.items() ]
        lookup		= dict()
        for i in range(len(new)) :
            for item in new[i].items() :
                lookup[item[0]] = item[1]
        #
        spec_group_class = pd.DataFrame( lookup.items() , index=lookup.keys() ).iloc[:,[-1]]
        spec_group_class = spec_group_class.loc[df_.index.values]
        #
        spec_group_class .columns       = [ slab ]
        sgc_df                          = spec_group_class.copy()
        sgc_df.loc[:,lab1]              = df_.loc[:,lab1]
        cases = sorted(list(set(sgc_df.loc[:,slab].values.tolist())))

        def gfunc( x , cases ) :
            name = x.iloc[0,-1]
            mapvals = Counter( x.iloc[:,0].values )
            for idx in cases:
                if not idx in mapvals:
                    mapvals[idx] = 0
            y = [ mapvals[c] for c in cases ]
            x = pd.Series( y, index = cases )
            return ( x )

        sgcdf = sgc_df.groupby(lab1).apply( lambda x: gfunc( x, cases = cases ) )
        sgcdf .columns = [ rename_cats[c] if c in rename_cats else c for c in cases ]
        idxs  = sgcdf.index.values.tolist()

        if bSorted and index_order is None :
            idxs = [ s[1] for s in sorted([ (s,i) for s,i in zip( np.sum(sgcdf.values,1),sgcdf.index.values.tolist() ) ]) ]
        if not index_order is None :
            idxs = [ *index_order,*list( set(idxs)-set(index_order) ) ]

        metric_df = (acdf.T.apply(compositional_analysis)).T
        metric_df .columns = [ 'Beta' , 'Tau' , 'Gini' , 'Geni', 'TSI', '(n-1)/n' ]
        metric_df = metric_df.loc[df_.index,:]
        metric_df .loc[:,lab1]	= df_.loc[:,lab1]
        sort_order		= sgcdf.loc[idxs,:].index.values
        trail			= []
        ordered			= []
        I			= 0
        for idx in metric_df.index.values :
            if I>=len(sort_order) :
                trail.append(idx)
            elif metric_df.loc[idx,lab1] == sort_order[I] :
                ordered.append(idx)
                I += 1
            else :
                trail.append(idx)
        morder = [ *ordered , *trail ]
        if len(morder) == len(metric_df):
            print ('SUCCESS!')
        return ( metric_df .loc[morder,:] , sgcdf.loc[idxs,:] )

def calculate_fisher_for_cluster_groups ( df:pd.DataFrame , label:str = None ,
                        gmtfile:str = None , pcfile:str = None , bVerbose:bool=True , bShallow:bool = False ,
                        test_type:str='fisher' , bUseAlexaElim:bool=False ,
                        significance_level:float = None , alternative:str='greater' ) -> dict :
    return( calculate_for_cluster_groups ( df=df , label = label ,
                        gmtfile = gmtfile , pcfile = pcfile , bVerbose=bVerbose , bShallow = bShallow ,
                        test_type=test_type ,  bUseAlexaElim=bUseAlexaElim,
                        significance_level = significance_level , alternative=alternative ) )

def calculate_for_cluster_groups ( df:pd.DataFrame , label:str = None ,
                        gmtfile:str = None , pcfile:str = None , bVerbose:bool=True , bShallow:bool = False ,
                        test_type:str='hypergeometric' , bUseAlexaElim:bool=False , 
                        bNoMasking:bool=False , bOnlyMaskSignificant:bool=False , item_sep:str=',' ,
                        group_identifier:str = 'R-HSA' ,
                        significance_level:float = None , alternative:str='greater' ) -> dict :
    if label is None :
        print ( "ERROR: YOU MUST SPECIFY A CLUSTER GROUPING LABEL" )
        exit ( 1 )
    if gmtfile is None or pcfile is None :
        print ( "ERROR: YOU MUST SPECIFY A INDEX GROUPING FILE" )
        exit ( 1 )
    all_indices = df.index.values.tolist()

    from impetuous.special import unpack
    import impetuous.hierarchical as imph
    dag_df , tree = imph .create_dag_representation_df ( pathway_file = gmtfile , pcfile = pcfile , identifier = group_identifier , item_sep = item_sep )
    rootid = tree.get_root_id()
    adf = df.groupby( label ).apply(lambda x:'|'.join(x.index.values.tolist()))
    dag_maps = dict()
    for idx in adf.index :
        sigids  = adf.loc[idx].replace('\n','').replace('\t','').split('|')
        tdf     = pd.DataFrame( [ 1.0 for v in all_indices] ,index=all_indices , columns=[idx] )
        tdf     .loc[ sigids ] = 0.001

        hdf = imph.HierarchicalEnrichment ( tdf , dag_df ,
                dag_level_label = 'DAG,level' , ancestors_id_label = 'DAG,ancestors' ,
                threshold = 0.05 , p_label = idx , analyte_name_label = 'analytes' ,
                item_delimiter = item_sep , alexa_elim = bUseAlexaElim , alternative = alternative ,
                test_type = test_type, bNoMasking=bNoMasking , bOnlyMarkSignificant=bOnlyMaskSignificant
        )
        hdf = hdf.sort_values( by='Hierarchical,p' )
        lookup = { hi:len( [l for l in v.split(item_sep) if len(l)>0] ) for v,hi in zip(hdf.loc[:,'Included analytes,ids'].values,hdf.index.values) }
        lookup[rootid] = len( sigids )
        if not significance_level is None :
            hdf = hdf.iloc[ hdf.loc[:,'Hierarchical,p'].values<=significance_level , : ]
        names           = hdf.index.values
        pvals           = hdf.loc[:,'Hierarchical,p'].values
        ancestors       = hdf.loc[:,'DAG,ancestors'].values
        parents         = dag_df.loc[ hdf.index , 'DAG,parent' ].values
        df_ = pd.DataFrame( [parents,pvals,hdf.loc[:,'description'].values,names] ).T
        df_ .columns    = ['parent','p-value','description','name']
        df_ .index      = df_.loc[:,'name'].values.tolist()
        add_idxs        = list ( set( df_.loc[:,'parent'].values.tolist() ) - set( df_.index.values.tolist() ) )
        tddf = dag_df.loc[ add_idxs,['Hierarchical,p','description'] ].copy()
        tddf .loc[:,'parent'] = [ rootid if rootid!=i else "" for i in add_idxs ]
        tddf .loc[:,'name'] = add_idxs
        tddf = tddf.rename(columns={'Hierarchical,p':'p-value'})
        df_  = pd.concat( [df_,tddf] )
        df_ .loc[:,'N_intersect'] = [ lookup[id_] for id_ in df_.index.values]
        dag_maps[ idx ] = df_
        if bVerbose :
            print ( df_ )
    """
    import plotly.express as px
    fig     = px.treemap( df_    ,
        names   = 'name'        ,
        parents = 'parent'      ,
        color   = 'p-value'     ,
        color_continuous_scale = 'RdBu' ,
        hover_data = ['description','N_intersect'] ,
    )
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()
    """
    return ( dag_maps )

from biocartograph.composition import composition_leading_label_and_metric
#
def get_annotation_function(	header_str:str = '../results/DMHMSY_Fri_Jun__2_09_15_35_2023_'	,
				auto_annotated_df:pd.DataFrame		= None 			,
				significance_level:float                = 0.1			,
				enrichment_file_pattern:str     = "(HEADER)treemap_c(CLUSTID).tsv" ) -> list[str] :
    #
    # ADD IN TOP ENRICHMENT AS FUNCTION
    function = []
    set_all_files = set( os.listdir( '/'.join(header_str.split('/')[:-1]) ) )
    for cid in auto_annotated_df .index.values.tolist() :
        filename        = enrichment_file_pattern.replace('(HEADER)',header_str).replace('(CLUSTID)',str(cid))
        fn              = filename.split('/')[-1]
        if not fn in set_all_files :
            function.append( 'Unknown' )
            continue
        df              = pd.read_csv(filename,sep='\t',index_col=0)
        if len(df) > 0 :
            if df.iloc[ np.argmin( df.loc[:,'p-value'].values ) , : ].loc['p-value'] < significance_level :
                function .append( df.iloc[ np.argmin( df.loc[:,'p-value'].values ) , : ].loc['description'] )
                continue
        function .append( 'Unknown' )
    return ( function )


def auto_annotate_clusters (	header_str:str = '../results/DMHMSY_Fri_Jun__2_09_15_35_2023_'	,
				alignment_information:str       	= "pcas_df.tsv"		,
				cluster_information:str         	= "resdf_f.tsv"		,
				final_cluster_label:str         	= "cids.max"		,
				significance_level:float		= 0.1			,
				enrichment_results_file_pattern:list[str] 	= ["(HEADER)treemap_c(CLUSTID).tsv"] ) -> pd.DataFrame :
    #
    df_pca =        pd.read_csv( header_str + alignment_information , sep='\t', index_col=0 )
    df_clu =        pd.read_csv( header_str + cluster_information   , sep='\t', index_col=0 )
    #
    merged_info = pd.concat ( [	df_clu .loc[:,[final_cluster_label]].T	,
				df_pca .loc[:,[ c for c in df_pca.columns if 'Owner' in c ]].T,
			 ]).T.rename( columns = { final_cluster_label:'cluster' } )
    merged_info.columns	= [ v.replace('Owner','PCA.owner') for v in  merged_info.columns.values ]
    consolidate		= [ c for c in merged_info.columns if 'owner' in c ]
    #
    def reformat_local( x , consolidate ) -> pd.Series :
        rv = [ len(x) ]
        labs = ['N']
        for console in consolidate :
            i = 0
            for w in composition_leading_label_and_metric( Counter(x.loc[ :, console ].values) ).values()  :
                rv  .append( w )
                postfix = ''
                if i==0 :
                    postfix = ',Tau'
                labs.append( console.replace( 'PCA.owner','Specificity' ) + postfix )
                i += 1
        return ( pd.Series(rv,index=labs) )
    #
    from collections import Counter
    #
    # AUTO ANNOTATIONS
    if len( consolidate ) > 1 :
        auto_annotated_df = merged_info.groupby('cluster').apply( lambda x: reformat_local(x,consolidate) )
    else :
        auto_annotated_df = merged_info.groupby('cluster').apply(lambda x: pd.Series( [ len(x) ,
                *[ v for v in composition_leading_label_and_metric(Counter(x.loc[ :, consolidate[0] ].values)).values() ] ] ,
                        index = [ 'N' , 'Reliability-Spec,Tau' , 'Specificity' ] ) )
    #
    # ADD IN TOP ENRICHMENT
    for enrichment_file_pattern,i_ in zip( enrichment_results_file_pattern , range(len(enrichment_results_file_pattern)) ) :
        function = get_annotation_function(	header_str 			= header_str			,
						auto_annotated_df		= auto_annotated_df		,
                                		significance_level 		= significance_level 		,
                                		enrichment_file_pattern		= enrichment_file_pattern	)
        auto_annotated_df.loc[:,'Function,'+str(i_)] = function

    return ( auto_annotated_df )


def benchmark_group_volcano_metrics ( group_df:pd.DataFrame , journal_df:pd.DataFrame ,
                        major_group_label:str = None , what_thing:str = None , bRanked:bool=False ,
                        nsamples:int = None, bVerbose:bool = False , bgname:str = 'Background' ,
                        clean_label = lambda x : x.replace(' ','_').replace('/','Or').replace('\\','').replace('-','') ) -> pd.DataFrame :

    from biocartograph.special import calculate_volcano_df
    from impetuous.quantification import qvalues
    df          = group_df
    if major_group_label is None :
        print ( "WARNING CANNOT CONTINUE. NEED major_group_label" )
        exit(1)
    jdf         = journal_df.loc[ major_group_label , : ]
    check_cols  = journal_df.columns.values.tolist()
    if not what_thing is None :
        lookup = { k:v for k,v in [ tuple( ( col , what_thing if clean_label(what_thing) in clean_label(label) else bgname )) for label,col in zip( jdf.values
,jdf.index.values) ] }
    else :
        print ( "WARNING CANNOT CONTINUE. NEED what_thing" )
        exit(1)
    vals_df     = df.loc[:,check_cols].apply(pd.to_numeric).dropna(axis=1)
    vals_df .loc['Regulation']  = [ lookup[c] for c in vals_df.columns.values ]
    fc_levels  = list( set(vals_df .loc['Regulation'])-set([bgname]) )
    fc_levels  .append( bgname )
    if len ( fc_levels ) != 2 :
        print ( "WARNING NOT WELL DEFINED TEST", fc_levels )
        exit(1)
    volcano_df = calculate_volcano_df ( vals_df , levels = fc_levels , what = 'Regulation' , bLog2 = True , bRanked = bRanked )
    volcano_df .loc[:,'id']             = df.index.values
    volcano_df .loc[:,'q-value']	= [ q[0] for q in qvalues( volcano_df.loc[:,'p-value'].values ) ]
    volcano_df .loc[:,'-log10 q-value'] = [ -1*np.log10(v) for v in volcano_df.loc[:,'q-value'].values ]
    return ( volcano_df )


def benchmark_group_expression_with_univariate_foldchange ( group_df:pd.DataFrame , journal_df:pd.DataFrame ,
                        major_group_label:str = None , what_thing:str = None , bRanked:bool=False ,
			nsamples:int = None, bVerbose:bool = False , bgname:str = 'Background' ,
			clean_label = lambda x : x.replace(' ','_').replace('/','Or').replace('\\','').replace('-','') ) -> pd.DataFrame :

    from biocartograph.special import calculate_volcano_df

    df		= group_df
    if major_group_label is None :
        print ( "WARNING CANNOT CONTINUE. NEED major_group_label" )
        exit(1)
    jdf		= journal_df.loc[ major_group_label , : ]
    check_cols	= journal_df.columns.values.tolist()
    if not what_thing is None :
        lookup = { k:v for k,v in [ tuple( ( col , what_thing if clean_label(what_thing) in clean_label(label) else bgname )) for label,col in zip( jdf.values,jdf.index.values) ] }
    else :
        print ( "WARNING CANNOT CONTINUE. NEED what_thing" )
        exit(1)
    vals_df     = df.loc[:,check_cols].apply(pd.to_numeric).dropna(axis=1)
    vals_df .loc['Regulation']  = [ lookup[c] for c in vals_df.columns.values ]
    fc_levels  = list( set(vals_df .loc['Regulation'])-set([bgname]) )
    fc_levels  .append( bgname )
    if len ( fc_levels ) != 2 :
        print ( "WARNING NOT WELL DEFINED TEST", fc_levels )
        exit(1)
    volcano_df = calculate_volcano_df ( vals_df , levels = fc_levels , what = 'Regulation' , bLog2 = True , bRanked = bRanked )
    volcano_df .loc[:,'Hierarchical,q'] = df.loc[:,'Hierarchical,q'].values
    volcano_df .loc[:,'-log10 hier-q' ] = -np.log10( volcano_df.loc[:,'Hierarchical,q'] )
    volcano_df .loc[:,'description']    = df.loc[:,'description']
    volcano_df .loc[:,'id']             = df.index.values
    sig_level  = 1.0 / nsamples if not nsamples is None else 0.05
    atmost_q   = sorted ( df.loc[:,'Hierarchical,q'].values )[nGroups]
    if atmost_q < sig_level :
        sig_level = atmost_q
    if bVerbose :
        print ( sig_level )
    volcano_df .loc[:,'color']          = [ 'red' if v <=sig_level  else 'gray' for v in df.loc[:,'Hierarchical,q'].values ]
    return ( volcano_df )
#
# GFA LIKE
def from_multivariate_group_factors (	analyte_df:pd.DataFrame ,
					journal_df:pd.DataFrame ,
                                	label:str , gmtfile:str , pcfile:str = None ,
                                        group_identifier:str='R-HSA' , item_delimiter:str=',' ,
					formula:str = None , block_formula:str = '' ,
					bUnivariateInstanceProjected:bool = False , bVerbose:bool=False , bPassOnVerbosity:bool=False ,
					control_group:str = None ) -> dict :
    #
    def use_formula ( formula:str=None, label:str=None ) -> str :
        if formula is None :
            formula = 'Group~'+'C(' + label + ')'
        return ( formula )
    #
    if bVerbose and not formula is None :
        print ( 'ANALYZING : ', formula )
    #
    journal_df_ = journal_df
    calculate_pairs = [ ]
    if bUnivariateInstanceProjected :
        for instance_label in set(journal_df.loc[label].values) :
            if 'str' in str(type(control_group)) :
                if instance_label == control_group :
                    continue
                journal_df_.loc[instance_label] = [ instance_label if (v == instance_label) else ( control_group if (v==control_group)  else 'Other' ) for v in journal_df.loc[label].values ]
                calculate_pairs .append( [instance_label , instance_label , control_group ] )
            else :
                journal_df_.loc[instance_label] = [ instance_label if v == instance_label else 'Other' for v in journal_df.loc[label].values ]
                calculate_pairs .append( [instance_label , instance_label , None ] )
    #
    if  bUnivariateInstanceProjected or formula is None :
        calculate_pairs = [ [label , None , None ] , *calculate_pairs ]
        if bVerbose :
            print ( 'CONDUCTING ANALYSIS FOR' , calculate_pairs , len(calculate_pairs) )
    #
    from impetuous.quantification	import groupFactorAnalysisEnrichment	as gFArEnr
    from impetuous.hierarchical	import groupFactorHierarchicalEnrichment	as gFAhEnr
    all_results = dict()
    for instance_axis in calculate_pairs :
        instance_label = instance_axis[0]
        if bUnivariateInstanceProjected or formula is None :
            if bVerbose :
                print ( 'CALCULATING ENRICHMENT FOR' , instance_axis  )
            mainContrasts	= journal_df_.loc[ label , : ]
            bMainContrasts	= [ True for v in journal_df_.columns ]
            if not (instance_axis[1] is None) :
                mainContrasts	= [ v if v==instance_axis[1] else 'Other' for v in journal_df_.loc[instance_label].values ]
            if not (instance_axis[2] is None) and not ( instance_axis[1] is None ) :
                mainContrasts	= [ v for v in journal_df_.loc[instance_label].values if v==instance_axis[2] or v==instance_axis[1] ]
                bMainContrasts	= [ v==instance_axis[2] or v==instance_axis[1] for v in journal_df_.loc[instance_label].values ]
            jdf = journal_df_	.iloc [ : , bMainContrasts ]
            jdf .loc[instance_label.replace(' ','_')]	=	mainContrasts
            adf = analyte_df	 .loc [ : ,    jdf.columns ]
            used_formula = use_formula( label=instance_label.replace(' ','_')  )
        else :
            adf = analyte_df
            jdf = journal_df
            used_formula = formula
        #
        used_formula = used_formula + block_formula
        if bVerbose :
            print ( 'USING THE FORMULA: ', used_formula )
            print ( 'GROUP MODULATION OF THE VALUES: ',set( jdf.loc[instance_label].values ) )
        #
        if pcfile is None :
            results = gFArEnr ( analyte_df = adf , journal_df = jdf ,
				formula = used_formula ,
				grouping_file = gmtfile , bVerbose = bVerbose and bPassOnVerbosity )
        else :
            import impetuous.hierarchical as imph
            dag_df , tree = imph .create_dag_representation_df ( pathway_file = gmtfile , pcfile = pcfile ,
                                                                 identifier = group_identifier , item_sep = item_delimiter )
            results = gFAhEnr ( analyte_df = adf , journal_df = jdf ,
				formula = used_formula , dag_df = dag_df ,
				bVerbose = bVerbose and bPassOnVerbosity )
        all_results = { **all_results , **{ instance_label : tuple( (results, used_formula) ) } }
    return ( all_results )
