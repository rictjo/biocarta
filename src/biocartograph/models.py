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
from collections import Counter

class WildDjungle ( object ) :
    def __init__ ( self , distance_type:str = 'euclidean' , bRegressor:bool=False , bReturnDictionaries:bool=False ) :
        self.id_          :str	= ""
        self.description_ :str	= """A WRAPPER CLASS FOR A RANDOM FORERST CLASSSIFIER BUT FIRST EXPANDS MEASURES INTO DISTANCES"""
        self.model_label_ :str	= ""
        self.bRegressor:bool	= bRegressor
        self.bSimplePredict:bool = not bReturnDictionaries
        self.model_order_ :list	= None
        self.array_order_ :list	= None
        self.data_model_df	: pd.DataFrame	= None
        self.target_model_df	: pd.DataFrame	= None
        self.auxiliary_label_df	: pd.DataFrame	= None
        self.bDataCompleted	: bool	= False
        self.descriptive_df	: pd.DataFrame	= None
        self.bg_label_		: str	= 'Background'
        self.fpr_ : np.array	= None
        self.tpr_ : np.array	= None
        self.auc_ : float	= None
        self.cvs_ : np.array	= None
        self.bDidFit_:bool	= False
        import scipy.stats as scs
        from scipy.spatial.distance	import pdist
        from sklearn			import metrics
        from sklearn.ensemble		import RandomForestClassifier
        if self.bRegressor :
            from sklearn.ensemble       import RandomForestRegressor
        self .metrics		= metrics
        self .distance_type:str	= distance_type
        self .pdist		= lambda x : pdist(x,self.distance_type)
        self .RFC		= RandomForestClassifier
        if self.bRegressor :
            self.RFC		= RandomForestRegressor
        self .edge_importances_:np.array	= None
        self .edge_labels_:list[str]		= None
        self .computational_model_		= None # THE UNDERLYING CLASSIFIER MODEL
        self .model_funx	= tuple( ( lambda x:x/np.max(x) , lambda x: np.log(x+1) ) )
        self .func	= lambda x: np.sum(x)
        self .moms	= lambda x: np.array( [np.mean(x),np.std(x),scs.skew(x),-scs.kurtosis(x) , (np.mean(x)*np.std(x))**(1/3) ] )

    def synthesize_data_model ( self , selection:list , adf:pd.DataFrame , jdf:pd.DataFrame , model_label:str , alignment_label:str ) :
        self .model_label_	= model_label
        synth , descriptive	= [] , []
        #
        bdf			= adf.loc[ selection , : ]
        lookupi			= { s:i for s,i in zip(	selection	, range(len(selection)) ) }
        self.model_order_	= sorted ( selection )
        self.array_order_	= [ lookupi[ self.model_order_[k] ] for k in range(len(selection)) ]
        Nr			= int( len(self.model_order_)*(len ( self.model_order_ )-1)*0.5 )
        for c in adf.columns.values :
            model   = bdf.loc[ self.model_order_ , [c] ].apply(pd.to_numeric)
            dists   = self.synthesize( model.values )
            mos     = self.moms( dists )
            descriptive .append( tuple( (jdf.loc[alignment_label,c] if self.model_label_ in jdf.loc[alignment_label,c] else self.bg_label_,
					 self.func( dists ) , *self.moms(dists) )) )
            synth .append( tuple( (jdf.loc[alignment_label,c] if model_label in jdf.loc[alignment_label,c] else self.bg_label_ , *list( dists )  )) )
        res_df = pd.DataFrame( descriptive , columns=['L','V','M1','M2','M3','M4','MM'] )
        self .data_model_df	= pd.DataFrame(synth).iloc[:,1:1+Nr]
        self .target_model_df	= pd.DataFrame(synth).iloc[:,[0]]
        self .descriptive_df	= res_df
        self .bDataCompleted	= True
        self .bDidFit_		= False

    def synthesize ( self , absolute:np.array ) -> np.array :
        return ( self.model_funx[1]( self.pdist( self.model_funx[0](absolute) ) ) )

    def set_model_label ( self, model_label:str ) :
        self.model_label_ = model_label

    def set_bg_label ( self, bg_label:str) :
        self.bg_label_ = bg_label

    def get_model_name(self)->str:
        return ( str(self.model_label_) + ' | ' + str(self.bg_label_) )

    def fit ( self , X:np.array = None , y:np.array = None , binlabel:int = 1 , vertex_labels:list[str] = None ) :
        labels = vertex_labels
        if self.bDataCompleted and (X is None or y is None) :
            ''
        elif not X is None and not y is None :
            self.model_order_ = [ i for i in range(len(X)) ]
            R = []
            for j in range( np.shape(X)[1] ) :
                Z = self.synthesize ( X[:,j].reshape(-1,1) )
                R .append( Z )
            self.data_model_df = pd.DataFrame(R)
            bDone = False
            if self.bRegressor == False :
                if not self.model_label_ is None :
                    if self.model_label_ in set( y ) :
                        v = [ self.model_label_ if self.model_label_ in y_ else self.bg_label_ for y_ in y ]
                        bDone = True
                if not bDone :
                    if binlabel in set(y):
                        v = np.array( [ int(y_ == binlabel) for y_ in y ] )
                    else :
                        print ( 'PLEASE SPECIFY A USEFUL MDOEL LABEL USING .set_model_label(model_label:str) PRIOR TO RUNNING')
                        self.bDidFit_ = False
                        self.bDataCompleted = False
                        exit(1)
                    self.model_label_ = binlabel
            else :
                    v = np.array(y).reshape(-1)
            self.target_model_df	= pd.DataFrame( v )
            self.bDataCompleted = True
        else :
            self.bDataCompleted = False
            print ( 'HAS NO MODEL. RETRAIN THE CLASSIFIER WITH VIABLE INPUT' )
        self .computational_model_	= self.RFC( )
        self .computational_model_ .fit( X=self.data_model_df.values , y=self.target_model_df.values.reshape(-1) )
        self .bDidFit_ = True
        self .edge_importances_ = self .computational_model_.feature_importances_
        nL = np.shape(self.data_model_df.values)[1]
        print ( nL , np.shape(self.data_model_df.values) , np.shape(self.target_model_df.values.reshape(-1)) )
        if labels is None :
            labels = [ str(i+1) for i in range(nL) ]
        else :
            nL = len( labels )
        self.edge_labels_ = [ labels[i]+':'+labels[j]  for i in range(nL) for j in range(nL) if (i<=j and i!=j) ]

    def predict_single_ (self,Y) -> list :
        xvs_ = self.synthesize( Y.reshape(-1,1) ).reshape(1,-1)
        return ( [ {'infered'       : self.computational_model_.predict( xvs_ )[0] ,
                    'probabilities' : self.computational_model_.predict_proba( xvs_ )[0] } ] )

    def predict ( self , X ) -> list :
        if not self.bDidFit_ :
            self.fit()
        nm = np.shape( X )
        if 'panda' in str(type(X)).lower() or 'serie' in str(type(X)).lower() :
            if not self.model_order_ is None :
                if len(set(self.model_order_) - set(X.index.values.tolist()) ) == 0 :
                    X = X .loc[self.model_order_]
                elif np.isreal( np.sum(self.model_order_) ) :
                    X = X.iloc[self.model_order_]
            X = X.values
        elif 'array' in str(type(X)).lower() and not self.array_order_ is None and len(nm)>1 :
            X = X[ self.array_order_ ,: ]
        if len( nm ) > 1 :
            return ( [ self.predict(x_)[0] for x_ in X.T ] )
        else : # np.reshape( np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) ,newshape=(3,3) )
            xvs_ = self.synthesize( X.reshape(-1,1) ).reshape(1,-1)
            if self.bSimplePredict :
                return ( [ self.computational_model_.predict( xvs_ )[0] ] )
            if self.bRegressor :
                return ( [ { 'infered'          : self.computational_model_.predict( xvs_ )[0] } ] )
            else :
                return ( [ { 'infered'		: self.computational_model_.predict( xvs_ )[0] ,
			 'probabilities'	: self.computational_model_.predict_proba( xvs_ )[0] } ] )

    def generate_metrics ( self , y_true:list , y_proba:list , ipos:int= -1 , n_cv:int=5 ) -> dict :
        if y_proba is None or len(y_true) != len(y_proba) :
            print ( "UNEQUAL INPUT LENGTHS NOT SUPPORTED" )
        if 'panda' in str( type(y_true) ) .lower() :
            y_true = [ int(  str(self.model_label_) in str(y)  ) for y in y_true.values.reshape(-1) ]
        if len( y_proba[0] ) == len( y_proba[-1] ) and len( y_proba[0] )>1 :
            y_proba = [ y[ipos] for y in y_proba ]
        fpr_ , tpr_ , rest = self.metrics.roc_curve( y_true , y_proba )
        self .fpr_	= fpr_
        self .tpr_	= tpr_
        self .auc_	= np.trapz( tpr_,fpr_ )
        if not n_cv is None or not self.bDataCompleted :
            from sklearn.model_selection import cross_val_score
            self .cvs_ = cross_val_score(self.computational_model_,self.data_model_df,self.target_model_df.values.reshape(-1),cv=n_cv)
        else :
            self .cvs_ = np.array( ['Not evaluated'] , np.dobject )
        return ( {	'FPR'		: self.fpr_	, 'TPR'	: self.tpr_	,
			'Additonal'	: rest		, 'AUC'	: self.auc_ 	,
			'CV'		: self.cvs_ 	} )


class DjungleClassifier ( WildDjungle ) :
    pass

    def __init__ ( self , distance_type:str = 'euclidean' , bReturnDictionaries:bool=True ) :
        self.id_          :str  = ""
        self.description_ :str  = """A WRAPPER CLASS FOR A RANDOM FORERST CLASSSIFIER BUT FIRST EXPANDS MEASURES INTO DISTANCES"""
        self.model_label_ :str  = ""
        self.bRegressor:bool    = False
        self.bSimplePredict:bool = not bReturnDictionaries
        self.model_order_ :list = None
        self.array_order_ :list = None
        self.data_model_df      : pd.DataFrame  = None
        self.target_model_df    : pd.DataFrame  = None
        self.auxiliary_label_df : pd.DataFrame  = None
        self.bDataCompleted     : bool  = False
        self.descriptive_df     : pd.DataFrame  = None
        self.bg_label_          : str   = 'Background'
        self.fpr_ : np.array    = None
        self.tpr_ : np.array    = None
        self.auc_ : float       = None
        self.cvs_ : np.array    = None
        self.bDidFit_:bool      = False
        import scipy.stats as scs
        from scipy.spatial.distance     import pdist
        from sklearn                    import metrics
        from sklearn.ensemble           import RandomForestClassifier
        if self.bRegressor :
            from sklearn.ensemble       import RandomForestRegressor
        self .metrics           = metrics
        self .distance_type:str = distance_type
        self .pdist             = lambda x : pdist(x,self.distance_type)
        self .RFC               = RandomForestClassifier
        if self.bRegressor :
            self.RFC            = RandomForestRegressor
        self .edge_importances_:np.array        = None
        self .edge_labels_:list[str]            = None
        self .computational_model_              = None # THE UNDERLYING CLASSIFIER MODEL
        self .model_funx        = tuple( ( lambda x:x/np.max(x) , lambda x: np.log(x+1) ) )
        self .func      = lambda x: np.sum(x)
        self .moms      = lambda x: np.array( [np.mean(x),np.std(x),scs.skew(x),-scs.kurtosis(x) , (np.mean(x)*np.std(x))**(1/3) ] )


class DjungleRegressor ( WildDjungle ) :
    pass

    def __init__ ( self , distance_type:str = 'euclidean' , bReturnDictionaries:bool=False ) :
        self.id_          :str  = ""
        self.description_ :str  = """A WRAPPER CLASS FOR A RANDOM FORERST CLASSSIFIER BUT FIRST EXPANDS MEASURES INTO DISTANCES"""
        self.model_label_ :str  = ""
        self.bRegressor:bool    = True
        self.bSimplePredict:bool = not bReturnDictionaries
        self.model_order_ :list = None
        self.array_order_ :list = None
        self.data_model_df      : pd.DataFrame  = None
        self.target_model_df    : pd.DataFrame  = None
        self.auxiliary_label_df : pd.DataFrame  = None
        self.bDataCompleted     : bool  = False
        self.descriptive_df     : pd.DataFrame  = None
        self.bg_label_          : str   = 'Background'
        self.fpr_ : np.array    = None
        self.tpr_ : np.array    = None
        self.auc_ : float       = None
        self.cvs_ : np.array    = None
        self.bDidFit_:bool      = False
        import scipy.stats as scs
        from scipy.spatial.distance     import pdist
        from sklearn                    import metrics
        from sklearn.ensemble           import RandomForestClassifier
        if self.bRegressor :
            from sklearn.ensemble       import RandomForestRegressor
        self .metrics           = metrics
        self .distance_type:str = distance_type
        self .pdist             = lambda x : pdist(x,self.distance_type)
        self .RFC               = RandomForestClassifier
        if self.bRegressor :
            self.RFC            = RandomForestRegressor
        self .edge_importances_:np.array        = None
        self .edge_labels_:list[str]            = None
        self .computational_model_              = None # THE UNDERLYING CLASSIFIER MODEL
        self .model_funx        = tuple( ( lambda x:x/np.max(x) , lambda x: np.log(x+1) ) )
        self .func      = lambda x: np.sum(x)
        self .moms      = lambda x: np.array( [np.mean(x),np.std(x),scs.skew(x),-scs.kurtosis(x) , (np.mean(x)*np.std(x))**(1/3) ] )
