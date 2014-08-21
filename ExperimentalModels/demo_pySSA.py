# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:19:48 2014

@author: john abel

"""

import numpy  as np
import casadi as cs
import pdb
import stochkit_resources as stk
import modelbuilder as mb
import pylab as pl
import circadiantoolbox_raw as ctb
import Bioluminescence as bl

EqCount = 11
ParamCount = 46
modelversion='deg_sync_v9_0_stochkit'

#initial values
y0in = np.array([ 0.09909523,  0.70371313,  0.2269922 ,  0.10408456,  0.00490967,
        0.86826377,  0.89688085,  0.06720938,  0.42133251,  0.00728958,
        0.47681956])

#known period (not necessary for running but useful for analysis sometimes)
period = 23.7000

#omega parameter for stochkit
vol=200

#parameter values            
param=[0.2846151688657202 , 0.232000177516616 , 0.088617203761593 , 0.30425468        , 0.210097869        , 
       0.4353107703541283 , 1.003506668772474 , 1.088997860405459 , 0.0114281138      , 1.37671691         , 
       2.6708076060464903 , 0.034139448       , 2.679624716511808 , 0.769392535473404 , 2.54809178         , 
       0.0770156091097623 , 0.305050587159186 , 0.0636139454      , 0.102828479472142 , 0.0021722217886776 , 
       3.4119930083042749 , 0.313135234185038 , 0.129134295035583 , 0.086393910969617 , 0.1845394740887122 , 
       0.1918543699832282 , 2.93509002        , 0.668784664       , 1.08399453        , 0.368097886        , 
       1.1283479292931928 , 0.305037169       , 0.530015145234027 , 0.317905521992663 , 0.3178454269093350 , 
       3.1683607          , 0.531341137364938 , 0.807082897       , 0.251529761689481 , 0.1805825385998701 , 
       1.418566520274632  , 0.835185094       , 0.376214021       , 0.285090232       , 0.27563398         ,
       1.113098655804457 ]

def StochModel():

    #==================================================================
    # Stochastic Model Setup
    #==================================================================
    
    print 'Now converting model to StochKit XML format...'
    
    #Converts concentration to population for initial values
    y0in_stoch = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = ['p', 'c1', 'c2', 'vip', 'P', 'C1', 'C2', 'eVIP', 'C1P', 'C2P', 'CREB']
                    
    param_array   = ['vtpr' , 'vtc1r' , 'vtc2r'  , 'knpr'   , 'kncr'   , 
                   'vdp'   , 'vdc1'  , 'vdc2'  , 'kdp'    , 'kdc'    ,
                   'vdP'   , 'kdP'   , 'vdC1'  , 'vdC2'   , 'kdC'    , 
                   'vdC1n' , 'vdC2n' , 'kdCn'  , 'vaCP'   , 'vdCP'   , 'ktlnp',
                   'vtpp'  , 'vtc1p' , 'vtc2p' , 'vtvp'   , 'vtvr'   , 
                   'knpp'  , 'kncp'  , 'knvp'  , 'knvr'   , 'vdv'    ,
                   'kdv'   , 'vdVIP' , 'kdVIP' , 'vgpcr'  , 'kgpcr'  , 
                   'vdCREB', 'kdCREB', 'ktlnv' , 'vdpka'  , 'vgpka'  , 
                   'kdpka' , 'kgpka' , 'kdc1'  , 'kdc2'   , 'ktlnc']
    
    #duplicated for names later
    state_names=species_array[:]
    param_names=param_array[:]
    
    #Names model
    SSAmodel = stk.StochKitModel(name=modelversion)
    #SSAmodel.units='concentration'
    
    #creates SSAmodel class object
    SSA_builder = mb.SSA_builder(species_array,param_array,y0in_stoch,param,SSAmodel,vol)
    
    # REACTIONS

    #per mRNA

    SSA_builder.SSA_MM('per mRNA activation','vtpp',km=['knpp'],Prod=['p'],Act=['CREB'])
    SSA_builder.SSA_MM('per mRNA repression','vtpr',km=['knpr'],Prod=['p'],Rep=['C1P','C2P'])
    SSA_builder.SSA_MM('per mRNA degradation','vdp',km=['kdp'],Rct=['p'])


    #cry1 mRNA
    SSA_builder.SSA_MM('c1 mRNA activation','vtc1p',km=['kncp'],Prod=['c1'],Act=['CREB'])
    SSA_builder.SSA_MM('c1 mRNA repression','vtc1r',km=['kncr'],Prod=['c1'],Rep=['C1P','C2P'])
    SSA_builder.SSA_MM('c1 mRNA degradation','vdc1',km=['kdc'],Rct=['c1'])
    
    #cry2 mRNA
    SSA_builder.SSA_MM('c2 mRNA activation','vtc2p',km=['kncp'],Prod=['c2'],Act=['CREB'])
    SSA_builder.SSA_MM('c2 mRNA repression','vtc2r',km=['kncr'],Prod=['c2'],Rep=['C1P','C2P'])
    SSA_builder.SSA_MM('c2 mRNA degradation','vdc2',km=['kdc'],Rct=['c2'])
    
    #vip mRNA
    SSA_builder.SSA_MM('vip mRNA activation','vtvp',km=['knvp'],Prod=['vip'],Act=['CREB'])
    SSA_builder.SSA_MM('vip mRNA repression','vtvr',km=['knvr'],Prod=['vip'],Rep=['C1P','C2P'])
    SSA_builder.SSA_MM('vip mRNA degradation','vdv',km=['kdv'],Rct=['vip'])
    
    #CRY1, CRY2, PER, VIP creation and degradation
    SSA_builder.SSA_MA_tln('PER translation' ,'P'   ,'ktlnp','p')
    SSA_builder.SSA_MA_tln('CRY1 translation','C1'  ,'ktlnc','c1')
    SSA_builder.SSA_MA_tln('CRY2 translation','C2'  ,'ktlnc','c2')
    SSA_builder.SSA_MA_tln('VIP translation' ,'eVIP','ktlnv','vip')
    
    SSA_builder.SSA_MM('PER degradation','vdP',km=['kdP'],Rct=['P'])
    SSA_builder.SSA_MM('C1 degradation','vdC1',km=['kdC'],Rct=['C1'])
    SSA_builder.SSA_MM('C2 degradation','vdC2',km=['kdC'],Rct=['C2'])
    SSA_builder.SSA_MA_deg('eVIP degradation','eVIP','kdVIP')
    
    #CRY1 CRY2 complexing
    SSA_builder.SSA_MA_complex('CRY1-P complex','C1','P','C1P','vaCP','vdCP')
    SSA_builder.SSA_MA_complex('CRY2-P complex','C2','P','C2P','vaCP','vdCP')
    SSA_builder.SSA_MM('C1P degradation','vdC1n',km=['kdCn'],Rct=['C1P','C2P'])
    SSA_builder.SSA_MM('C2P degradation','vdC2n',km=['kdCn'],Rct=['C2P','C1P'])
    
    #VIP/CREB Pathway
    SSA_builder.SSA_MM('CREB formation','vgpka',km=['kgpka'],Prod=['CREB'],Act=['eVIP'])
    SSA_builder.SSA_MM('CREB degradation','vdCREB',km=['kdCREB'],Rct=['CREB'])
    
    
    # END REACTIONS
    #stringSSAmodel = SSAmodel.serialize()
    #print stringSSAmodel

    return SSAmodel,state_names,param_names





def main():
    
    #Creates SSA version of model.
    SSAmodel,state_names,param_names=StochModel()
    
    #calls and runs stochkit
    trajectories = stk.stochkit(SSAmodel,job_id='test',t=75,number_of_trajectories=100,increment=0.1)
    
    #evaluation bit
    StochEval = stk.StochEval(trajectories,state_names,param_names,vol)
    StochEval.PlotAvg('p',color='blue')
    pl.show()
    pdb.set_trace()

if __name__ == "__main__":
    main()  















