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
import pandas as pd
import numpy  as np
import typing

from impetuous.convert import NodeGraph, Node
from biocartograph.special import create_color, DrawGraphText, solve_border_salesman, create_hilbertmap
from impetuous.special import hc_d2r , hc_r2d , hc_assign_lin_nn , hc_assign_lin_nnn

unique_list         = lambda Y : [ list(li) for li in list(set( [ tuple((z for z in y)) for y in Y ] )) ]
make_hex_color      = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)
invert_color        = lambda color : make_hex_color( [  255 - int('0x'+color[1:3],0) ,
                                        255 - int('0x'+color[3:5],0) ,
                                        255 - int('0x'+color[5:] ,0) ] )
unordered_remove    = lambda Y,Z :  [ list(li) for li in list( set( [ tuple((z for z in y)) for y in Y ])\
                                                             - set( [ tuple((x for x in z)) for z in Z ]) ) ]
list_is_in_list     = lambda Y,Z :  len( set([ tuple((z for z in y)) for y in Y ]) - set([ tuple((x for x in z)) for z in Z ]) ) == 0

#
def show_hilbertmap_simple ( dR:dict , bAddShadow:bool = False , bShowCells:bool=False  ) :
    import matplotlib.patches   as patches
    import matplotlib.pyplot    as plt
    import matplotlib.text      as text

    fig , ax    = plt.subplots(1)
    for label in dR.keys() :
        if 'P data' in label or 'NearestN' in label :
            continue
        group_label = label
        X           = dR[ label ][1:]
        color       = dR[ label ][0]
        weight      = 1
        center      = np.mean( X ,0 )
        plt.plot( center[0] , center[1] , marker='.' , c=color )
        edgecolor   = 'black' if bShowCells else color
        facecolor   = color
        for x in X :
            ax.add_artist( patches.Rectangle( x , width=1 , height=1 , edgecolor=edgecolor , facecolor=facecolor ) )
        tcolor = invert_color( color )
        center = np.median(X,0)

        n    = np.max(X,0)[0]
        xpos = np.median(X,0)[0]
        if xpos + len(label)/6 > n :
            xpos -= ( len(label) / 8 )
            center[0] = xpos

        # INVERT THE COLOR TO MAKE THE TEXT VISIBLE
        if bAddShadow :
            ax.add_artist( text.Text( x=xpos-0.1 , y=center[1] , text=label , fontsize=10.3 ,
                    fontfamily='sans-serif' , weight='bold' , color = 'black' ) )
        ax.add_artist( text.Text( x=xpos , y=center[1] , text=label , fontsize=10 ,
                    fontfamily='sans-serif' , weight='bold', fontstyle='normal' , color = tcolor ) )
    plt.show()

def show_hilbertmap_polygons( dR:dict , color_label:str=None , text_pos_label:str = None ,
		bHideBorders:bool = False , bExpanded:bool=True , bInputDataIsPolygoned:bool = False ,
		bHideCenters:bool = True  , bAddLabels:bool=False ,
		bPlaceLabelsFarLeft:bool = True ) :
    import matplotlib.pyplot    as plt
    import matplotlib.text      as text
    import matplotlib.patches   as patches
    fig , ax    = plt.subplots(1)
    for label in dR.keys() :
        if 'P data' in label or 'NearestN' in label :
            continue
        group_label = label
        color   = dR[group_label][0] if color_label is None else dR[color_label][label]
        coords  = dR[group_label][1:]
        if bInputDataIsPolygoned :
            polygon = np.array(coords[:-1][0])
            center  = coords[ -1]
        else :
            expanded_coords = []
            for c in coords :
                expanded_coords.append( np.array(c) )
                if bExpanded :
                    expanded_coords.append( np.array(c) + np.array([ 1 , 0 ]) )
                    expanded_coords.append( np.array(c) + np.array([ 1 , 1 ]) )
                    expanded_coords.append( np.array(c) + np.array([ 0 , 1 ]) )
            expanded_coords = np.array( unique_list( expanded_coords ) )
            #
            from scipy.spatial.distance import pdist,squareform
            border = np.sum( squareform(pdist(expanded_coords)) <= 1 , 1 ) <= 4
            border_coords = np.array([x for x,b in zip(expanded_coords,border) if b ])
            center = np.median(expanded_coords,0)
            polygon = solve_border_salesman( border_coords )
        #
        plt.plot( center[0] , center[1] , marker='.' , c='k' if not bHideCenters else color )
        ax.add_artist(	patches.Polygon( polygon ,
			edgecolor = color if bHideBorders else 'black' ,
			facecolor = color ) )
        if bAddLabels :
            n = np.max(polygon,0)[0]
            xpos = np.median( polygon , 0 )[0]
            if xpos + len(label)/6 > n :
                xpos -= ( len(label) / 8 )
                if xpos<0 :
                    xpos = 0.5
                center[0] = xpos
            if bPlaceLabelsFarLeft :
                center[0] = np.min(polygon,0)[0] + 0.5
            if not text_pos_label is None :
                center = dR[text_pos_label][label]
            tcolor = invert_color( color )
            ax.add_artist( text.Text( x=center[0] , y=center[1] , text=label , fontsize=10 ,
                    fontfamily='sans-serif' , weight='bold', fontstyle='normal' , color = tcolor ) )
    plt.show()

def return_hilbertmap_polygons( dR:dict , bExpanded:bool=True , color_label:str=None ) -> dict :
    polygons = dict()
    for label in dR.keys() :
        if 'P data' in label or 'NearestN' in label :
            continue
        group_label = label
        color   = dR[group_label][0] if color_label is None else dR[color_label][label]
        coords  = dR[group_label][1:]
        expanded_coords = []
        for c in coords :
            expanded_coords.append( np.array(c) )
            if bExpanded :
                expanded_coords.append( np.array(c) + np.array([ 1 , 0 ]) )
                expanded_coords.append( np.array(c) + np.array([ 1 , 1 ]) )
                expanded_coords.append( np.array(c) + np.array([ 0 , 1 ]) )
        expanded_coords = np.array( unique_list( expanded_coords ) )
        #
        from scipy.spatial.distance import pdist,squareform
        border = np.sum( squareform(pdist(expanded_coords)) <= 1 , 1 ) <= 4
        border_coords = np.array([x for x,b in zip(expanded_coords,border) if b ])
        center = np.median(expanded_coords,0)
        polygon = solve_border_salesman( border_coords )
        polygons[label] = [ color , polygon , center ]
    return ( polygons )


if __name__ == '__main__' :
    #
    nG_ = create_NodeGraph_object_from_treemap_file( '../bioc_results/DMHMSY_Fri_Feb__2_13_16_01_2024_treemap_c4.tsv' )
    #
    if False :
        print ( "THE JSON DATA" )
        print ( nG_.write_json() )
        print ( "THE LEAF NODES" )
        print ( nG_.retrieve_leaves( nG_.get_root_id() ) )
    #
    d:int =  0
    n:int = 32
    m:int =  n
    #
    dR = create_hilbertmap ( nG_			,
                quant_label = 'Significance'	, #'Significance', # quant_label = 'Area'
                search_type = 'breadth'		, # search_type = 'depth'
                n = n				)
    P  = dR[ 'P data' ]
    NN = dR[ 'NearestN' ]
    #
    show_hilbertmap_plygons( dR , bAddLabels=True )
    dP = return_hilbertmap_polygons( dR )
    show_hilbertmap_plygons( dP , bInputDataIsPolygoned=True , bAddLabels=True )
    show_hilbertmap_simple ( dR )
    #
    dgt = DrawGraphText(        color_label = 'Color' , area_label = 'Area',
                        celltext_label = 'Description' , font = 'Arial' )
    #
    dgt .create_gv_node_info( nG_.get_root_id() , nG_  )
    graphtext = dgt.return_story()
    #
    import pygraphviz as pgv
    G1 = pgv.AGraph( graphtext )
    G1 .layout()
    G1 .draw("file1.svg")
