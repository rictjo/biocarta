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
import numpy  as np
import pandas as pd
import typing

from collections import Counter

from impetuous.quantification import composition_absolute, compositional_analysis
#
def calculate_compositions (	adf:pd.DataFrame , jdf:pd.DataFrame , label:str,
				bAddPies:bool=True, composition_type:str = 'absolute' ) -> pd.DataFrame :
    cdf = None
    if composition_type == 'absolute'	:
        cdf                 = composition_absolute ( adf=adf , jdf=jdf , label=label )
    if composition_type == 'normed'	:
        cdf		= composition_absolute ( adf=adf.apply(lambda x:x/np.sum(x) ) , jdf=jdf , label=label )
    if composition_type == 'scaled' or composition_type=='fractional' :
        c2l             = Counter(jdf.loc[label])
        number_freq     = np.array( [ c2l[l] for l in jdf.loc[label] ] )
        if composition_type == 'scaled' :
            cdf             = composition_absolute ( adf=adf*number_freq , jdf=jdf , label=label )
        if composition_type == 'fractional' :
            cdf             = composition_absolute ( adf=adf/number_freq , jdf=jdf , label=label )
    if composition_type == 'equalized'	:
        print ('WARNING : HISTOGRAM EQUALIZING COMPOSITIONS' )
        from scipy.stats import rankdata
        def h_eq ( x ) :
            rx = rankdata ( x , 'average' )
            mx = np.max( rx )
            return ( rx / mx )
        cdf                 = composition_absolute ( adf=adf.apply(lambda x: h_eq(x)) , jdf=jdf , label=label )
    if cdf is None :
        print ( 'ASSIGNING ABSOLUTE COMPOSITIONS' )
        cdf                 = composition_absolute ( adf=adf , jdf=jdf , label=label )
    #
    composition_df      = cdf.T.apply(compositional_analysis).T
    composition_df .columns = ['Beta','Tau','Gini','Geni','TSI','FILLING']
    max_quant_df        = cdf.T.apply(lambda x: x.index.values[np.argmax(x)] )
    composition_df .loc[ : , 'Leading Quantification Label' ] = max_quant_df.values.tolist()
    if bAddPies :
        from impetuous.quantification import composition_piechart
        fractions_df    = composition_piechart( cdf )
        return ( pd.concat( [composition_df.T, fractions_df]).T )
    return ( composition_df )

def composition_leading_label_and_metric ( composition:dict , metric:int = 1 ) -> dict :
    items = list(composition.items())
    leading_label = items[ np.argmax( [ v[1] for v in items ]) ]
    metric_value = compositional_analysis( np.array([v for v in composition.values()]) )[metric]
    return ( { 'composition metric' : metric_value , 'leading label' : leading_label[0] } )

def dual_axis_compositions ( adf:pd.DataFrame , jdf:pd.DataFrame , aux_grouping:list[str] ,
			     label:str , bAddPies:bool = True , func:str = 'fractional') -> pd.DataFrame :
    #
    cdf = composition_absolute( adf,jdf,label )
    if not len(aux_grouping) == len(adf) :
        print ( "ERROR: CANNOT CONTINUE aux_grouping MUST BE AXIS=0 GROUPING LABELS" )
        print ( " not len(aux_grouping) == len(adf) " )
        exit(1)
    cdf             = cdf.apply( pd.to_numeric )
    if func == 'absolute' :
        ccdf            = composition_absolute ( cdf.T                              ,
				pd.DataFrame([aux_grouping] , index=['group'], columns=adf.index.values.tolist())  ,
				'group' ) .T
    if func == 'fractional' :
        ccdf            = composition_absolute ( cdf.apply(lambda x:x/np.sum(x) ).T ,
				pd.DataFrame([aux_grouping] , index=['group'], columns=adf.index.values.tolist() )  ,
				'group' ) .T
    composition_df = ccdf.T.apply(lambda x:x/np.sum(x) ).T
    metrics_df = ccdf.T.apply(compositional_analysis).T
    metrics_df .columns		= ['Beta','Tau','Gini','Geni','TSI','FILLING']
    return ( {'composition':composition_df,'metrics':metrics_df} )


def calculate_cluster_compositions ( adf , jdf , label , cluster_labels:list[str] , itype:int=0 ) -> dict :
    res = dual_axis_compositions( adf , jdf , label ,
                        func = ['fractional','absolute'][itype] ,
                        aux_grouping = cluster_labels )
    return ( res )

