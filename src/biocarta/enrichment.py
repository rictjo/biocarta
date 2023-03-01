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

def create_specificity_group_df ( acdf:pd.DataFrame , df_:pd.DataFrame , index_order:list=None ,
                                  label:str = '' , post_fix:str = '' , bSorted:bool=True , slab:str = 'Specificity Class' ,
                                  rename_cats:dict = { '0':'ND' , '1':'LS' , '2':'GE' , '3':'GR' , '4':'TR' } ) :
        from impetuous.quantification import group_classifications , composition_absolute, compositional_analysis
        from collections import Counter
        lab1 = label
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
			gmtfile:str = None , pcfile:str = None ,
                        bVerbose:bool = True , bShallow:bool = False ,
			significance_level:float = None ) -> dict :

    def recalculate_parent_depth( nidx , tree ) :
        path_info       = tree.search( root_id=nidx , linktype='ascendants', order='depth' )
        pathway         = path_info['path'][1:]
        parent          = pathway[-1]
        depth           = -1
        higher          = hdf.iloc[ hdf.loc[:,'DAG,level'].values < hdf.loc[ nidx , 'DAG,level' ] ].loc[:,'DAG,ancestors']
        for jdx in higher.index :
            what = set([ jdx, *higher.loc[jdx].split(',') ])
            if len( what & set(pathway[:-1]) )>0  :
                npath = tree.search(    root_id         = jdx ,
                                        linktype        = 'ascendants' ,
                                        order           = 'depth' )['path']
                for ipath in range(len(npath)) :
                    if npath[ipath] in set(pathway[:-1]) :
                        n_depth = len(npath)-ipath
                        if n_depth>depth :
                            parent	= npath[ipath]
                            if bShallow :
                                n_depth	= depth
                            else :
                                depth	= n_depth
                            break
        return ( parent, depth )

    if label is None :
        print ( "ERROR: YOU MUST SPECIFY A CLUSTER GROUPING LABEL" )
        exit ( 1 )
    if gmtfile is None or pcfile is None :
        print ( "ERROR: YOU MUST SPECIFY A INDEX GROUPING FILE" )
        exit ( 1 )
    all_indices = df.index.values.tolist()

    from impetuous.special import unpack
    import impetuous.hierarchical as imph
    dag_df , tree = imph .create_dag_representation_df ( pathway_file = gmtfile , pcfile = pcfile )
    rootid = tree.get_root_id()
    adf = df.groupby( label ).apply(lambda x:'|'.join(x.index.values.tolist()))
    dag_maps = dict()
    for idx in adf.index :
        sigids	= adf.loc[idx].replace('\n','').replace('\t','').split('|')
        tdf	= pd.DataFrame( [ 1.0 for v in all_indices] ,index=all_indices , columns=[idx] )
        tdf	.loc[ sigids ] = 0.001

        hdf = imph.HierarchicalEnrichment ( tdf , dag_df ,
		dag_level_label = 'DAG,level' , ancestors_id_label = 'DAG,ancestors' ,
		threshold = 0.05 , p_label = idx , analyte_name_label = 'analytes' ,
		item_delimiter = ',' , alexa_elim = False , alternative = 'two-sided'
        )
        hdf = hdf.sort_values( by='Hierarchical,p' )
        lookup = { hi:len( [l for l in v.split(',') if len(l)>0] ) for v,hi in zip(hdf.loc[:,'Included analytes,ids'].values,hdf.index.values) }
        lookup[rootid] = len( sigids )
        if not significance_level is None :
            hdf = hdf.iloc[ hdf.loc[:,'Hierarchical,p'].values<=significance_level , : ]
        names		= hdf.index.values
        pvals		= hdf.loc[:,'Hierarchical,p'].values
        ancestors	= hdf.loc[:,'DAG,ancestors'].values
        parents		= []
        for name in names :
            rs = recalculate_parent_depth( name , tree )
            parents.append(rs[0])
        df_ = pd.DataFrame( [parents,pvals,hdf.loc[:,'description'].values,names] ).T
        df_ .columns	= ['parent','p-value','description','name']
        df_ .index	= df_.loc[:,'name'].values.tolist()
        add_idxs 	= list ( set( df_.loc[:,'parent'].values.tolist() ) - set( df_.index.values.tolist() ) )
        tddf = dag_df.loc[ add_idxs,['Hierarchical,p','description'] ].copy()
        tddf .loc[:,'parent'] = [ rootid if rootid!=i else "" for i in add_idxs ]
        tddf .loc[:,'name'] = add_idxs
        tddf = tddf.rename(columns={'Hierarchical,p':'p-value'})
        df_  = pd.concat( [df_,tddf] )
        df_ .loc[:,'N_intersect'] = [ lookup[id_] for id_ in df_.index.values]
        dag_maps[ idx ]	= df_
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
