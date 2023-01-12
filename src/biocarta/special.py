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

def read_rds_distance_matrix ( filename = '../res1/distance/distances.rds' ) :
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from scipy.spatial.distance import pdist,squareform
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    return ( squareform(readRDS ('../res1/distance/distances.rds') ) )

def inplace_norm ( x:pd.DataFrame , n:int=10 , random_state:int=42 ) -> pd.DataFrame :
    from sklearn.preprocessing import quantile_transform
    x = x.T # WE NORMALISE OVER ROWS
    nvals = quantile_transform ( x.values[:-1] , random_state=random_state , copy=True )
    return ( pd.DataFrame( nvals , index=x.index.values[:-1] , columns=x.columns.values ).T )

def quantile_class_normalisation ( adf:pd.DataFrame , classes:list[str]=None ,
		 n:int=10 , random_state:int=42 ) -> pd.DataFrame :
    if classes is None :
        from scipy.stats import rankdata
        return ( adf.apply(lambda x:(rankdata(x.values,'average')-0.5)/len(set(x.values))) )
    else :
        adf.loc['QuantileClasses'] = classes
        return ( adf.T.groupby('QuantileClasses') .apply( lambda x : inplace_norm( x ,
		 n=n , random_state=random_state) ).T )
