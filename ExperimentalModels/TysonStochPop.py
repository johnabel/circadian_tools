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


#initialization
EqCount = 2
ParamCount = 7
modelversion='tysoncoupled'
xdim=8
ydim=8
y0in = np.array([ 0.65609071,  0.85088331])
period = 3.04
vol=100
#Fit found for parameters            
param=[2., 20., 1., 0.005, 0.05, 0.1, 1.]




def ODEModel():
    # Variable Assignments
    X = cs.ssym("X")
    Y = cs.ssym("Y")

    sys = cs.vertcat([X,Y]) # vector version of y
    
    # Parameter Assignments
    P  = cs.ssym("P")
    kt = cs.ssym("kt")
    kd = cs.ssym("kd")
    a0 = cs.ssym("a0")
    a1 = cs.ssym("a1")
    a2 = cs.ssym("a2")
    kdx= cs.ssym("kdx")
    
    paramset = cs.vertcat([P, kt, kd, a0, a1, a2, kdx])
                        
    
    #================================================================================
    # Model Equations
    #================================================================================
    
    ode = [[]]*EqCount

    # MRNA Species
    t = cs.ssym("t")
    
    ode = [[]]*EqCount
    ode[0] = 1 / (1 + Y**P) - kdx*X
    ode[1] = kt*X - kd*Y - Y/(a0 + a1*Y + a2*Y**2)
    
    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset), cs.daeOut(ode=ode))
                        
    fn.setOption("name","2state")
    return fn
    #==================================================================
    # Stochastic Model Portion
    #==================================================================
    
def SSAmodel(fn,y0):
    
    #parameter initialization
    xlen = xdim
    ylen = ydim
    
    #Converts concentration to population
    y0in_ssa = (vol*y0).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    y0in_pop = []
    
    #coupling section===========================
    for indx in range(xlen):
        for indy in range(ylen):
            index = '_'+str(indx)+'_'+str(indy)
            #loops to include all species, normally this is the only line needed without index
            species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]
            state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]  
                            
            y0in_pop = np.append(y0in_pop, y0in_ssa)       
    
    #===========================================
            
    param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    #Names model
    SSAmodel = stk.StochKitModel(name=modelversion)
    
    #creates SSAmodel class object
    SSA_builder = mb.SSA_builder(species_array,param_array,y0in_pop,param,SSAmodel,vol)
    

    #coupling section
    for indx in range(xlen):
        for indy in range(ylen):
            
            index = '_'+str(indx)+'_'+str(indy)
            
            
            
            # REACTIONS
    
            
            SSA_builder.SSA_tyson_x('x prod nonlinear term'+index,'X'+index,'Y'+index,'P')
            SSA_builder.SSA_MA_deg('x degradation'+index,'X'+index,'kdx')
            
            SSA_builder.SSA_MA_tln('y creation'+index, 'Y'+index, 'kt', 'X'+index)
            SSA_builder.SSA_MA_deg('y degradation, linear'+index,'Y'+index,'kd')
            SSA_builder.SSA_tyson_y('y nonlinear term'+index,'Y'+index,'a0','a1','a2')

    return SSAmodel,state_names,param_names
    
#pulse model
def PulseModel(fn,y0pulse):
    xlen = xdim
    ylen = ydim
    
    #Converts concentration to population
    y0in_pop_pulse = y0pulse.astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    
    parampulse=[2., 20., 1., 0.005, 0.05, 0.1, 0  ]
              #[P,  kt,  kd, a0,    a1,   a2,  kdx]
    
    #coupling section===========================
    for indx in range(xlen):
        for indy in range(ylen):
            index = '_'+str(indx)+'_'+str(indy)
            #loops to include all species, normally this is the only line needed without index
            species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]
            state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]  
    #===========================================
            
    param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    #Names model
    PulseModel = stk.StochKitModel(name=modelversion)
    
    #creates SSAmodel class object
    Pulse_builder = mb.SSA_builder(species_array,param_array,y0in_pop_pulse,parampulse,PulseModel,vol)
    

    #coupling section
    for indx in range(xlen):
        for indy in range(ylen):
            
            index = '_'+str(indx)+'_'+str(indy)
            
            
            
            # REACTIONS
    
            
            Pulse_builder.SSA_tyson_x('x prod nonlinear term'+index,'X'+index,'Y'+index,'P')
            Pulse_builder.SSA_MA_deg('x degradation'+index,'X'+index,'kdx')
            
            Pulse_builder.SSA_MA_tln('y creation'+index, 'Y'+index, 'kt', 'X'+index)
            Pulse_builder.SSA_MA_deg('y degradation, linear'+index,'Y'+index,'kd')
            Pulse_builder.SSA_tyson_y('y nonlinear term'+index,'Y'+index,'a0','a1','a2')
    
    
    return PulseModel
    
    
def PostModel(fn,y0post):
    xlen = xdim
    ylen = ydim
    
    #Converts concentration to population
    y0in_pop_post = y0post.astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    
    parampost=[2., 20., 1., 0.005, 0.05, 0.1, 1]
    
    #coupling section===========================
    for indx in range(xlen):
        for indy in range(ylen):
            index = '_'+str(indx)+'_'+str(indy)
            #loops to include all species, normally this is the only line needed without index
            species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]
            state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]  
    #===========================================
            
    param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    #Names model
    PostModel = stk.StochKitModel(name=modelversion)
    
    #creates SSAmodel class object
    Post_builder = mb.SSA_builder(species_array,param_array,y0in_pop_post,parampost,PostModel,vol)
    

    #coupling section
    for indx in range(xlen):
        for indy in range(ylen):
            
            index = '_'+str(indx)+'_'+str(indy)
            
            
            
            # REACTIONS
    
            
            Post_builder.SSA_tyson_x('x prod nonlinear term'+index,'X'+index,'Y'+index,'P')
            Post_builder.SSA_MA_deg('x degradation'+index,'X'+index,'kdx')
            
            Post_builder.SSA_MA_tln('y creation'+index, 'Y'+index, 'kt', 'X'+index)
            Post_builder.SSA_MA_deg('y degradation, linear'+index,'Y'+index,'kd')
            Post_builder.SSA_tyson_y('y nonlinear term'+index,'Y'+index,'a0','a1','a2')
    
    
    return PostModel

def Controlmodel(fn,y0):
    
    #parameter initialization
    xlen = xdim
    ylen = ydim
    
    #Converts concentration to population
    y0in_ssa = (vol*y0).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    y0in_pop = []
    
    #coupling section===========================
    for indx in range(xlen):
        for indy in range(ylen):
            index = '_'+str(indx)+'_'+str(indy)
            #loops to include all species, normally this is the only line needed without index
            species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]
            state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]  
                            
            y0in_pop = np.append(y0in_pop, y0in_ssa)       
    
    #===========================================
            
    param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    #Names model
    Controlmodel = stk.StochKitModel(name=modelversion)
    
    #creates SSAmodel class object
    Control_builder = mb.SSA_builder(species_array,param_array,y0in_pop,param,Controlmodel,vol)
    

    #coupling section
    for indx in range(xlen):
        for indy in range(ylen):
            
            index = '_'+str(indx)+'_'+str(indy)
            
            
            
            # REACTIONS
    
            
            Control_builder.SSA_tyson_x('x prod nonlinear term'+index,'X'+index,'Y'+index,'P')
            Control_builder.SSA_MA_deg('x degradation'+index,'X'+index,'kdx')
            
            Control_builder.SSA_MA_tln('y creation'+index, 'Y'+index, 'kt', 'X'+index)
            Control_builder.SSA_MA_deg('y degradation, linear'+index,'Y'+index,'kd')
            Control_builder.SSA_tyson_y('y nonlinear term'+index,'Y'+index,'a0','a1','a2')

    return Controlmodel

def main():
    
    #Creates SSA version of model.
    #ODEmodel,SSAmodel,state_names,param_names=FullModel(True)
    #pretrajectories = np.array(stk.stochkit(SSAmodel,job_id='prepulse',t=6,number_of_trajectories=1,increment=0.01))
    #pdb.set_trace()
    #pre = np.hstack(pretrajectories)
    #new initial values
    #y0pulse = pre[1:len(pre)-1,:]
    
    """
    PulseModel = PulseModel(ODEmodel,y0pulse)
    pulsetrajectories = np.array(stk.stochkit(PulseModel,job_id='pulse',t=0.2*3.04/(2*pi),number_of_trajectories=1,increment = 0.01))
    
    
    y0post = pulsetrajectories[:]
    
    PostModel = PostModel(ODEmodel,y0post)
    posttracjectories =stk.stochkit(PostModel,job_id='postpulse',t=13,number_of_trajectories=1,increment = 0.1)
    """
    pdb.set_trace()

if __name__ == "__main__":
    main()  















