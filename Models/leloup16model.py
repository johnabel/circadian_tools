# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:24:17 2014
Modified from PSJ

@author: john
"""
import casadi as cs
import numpy as np

# Sensitivities
abstol = 1e-11
reltol = 1e-9
maxnumsteps = 40000

# Constants and Equation Setup
EqCount = 16
ParamCount = 52

param = [0.4 , 0.2 , 0.4  , 0.2  , 0.4  , 0.2  , 0.5  , 0.1  , 0.7 ,
            0.6 , 2.2 , 0.01 , 0.01 , 0.01 , 0.01 , 0.12 , 0.3  , 0.1 ,
            0.1 , 0.4 , 0.4  , 0.31 , 0.12 , 1.6  , 0.6  , 2    , 4   ,
            0.5 , 0.6 , 0.4  , 0.4  , 0.1  , 0.1  , 0.3  , 0.1  , 0.5 ,
            0.4 , 0.2 , 0.1  ,  0.5  , 0.6  , 0.7  , 0.8  , 0.7 ,
            0.7 , 0.7 , 0.8  , 1    , 1.1  , 1    , 1.1  , 1.5]

y0in = [4.59407105,   2.86451577,   7.48884079,   1.05945104,
        8.08220288,   0.05748281,   0.70488658,   4.56530925,
        1.57704986,   0.24563968,   0.21626293,   2.01114509,
        0.92338126,   0.88380173,   0.31640511,   0.57828811]
        
period = 23.80801488

def model():

    #==================================================================
    #setup of symbolics
    #==================================================================
    
    p   = cs.SX.sym("p")
    c   = cs.SX.sym("c")
    b   = cs.SX.sym("b")
    PC   = cs.SX.sym("PC")
    CC   = cs.SX.sym("CC")
    PCP  = cs.SX.sym("PCP")
    CCP  = cs.SX.sym("CCP")
    PCC  = cs.SX.sym("PCC")
    PCN  = cs.SX.sym("PCN")
    PCCP = cs.SX.sym("PCCP")
    PCNP = cs.SX.sym("PCNP")
    BC   = cs.SX.sym("BC")
    BCP  = cs.SX.sym("BCP")
    BN   = cs.SX.sym("BN")
    BNP  = cs.SX.sym("BNP")
    IN   = cs.SX.sym("IN")
    
    sys = cs.vertcat([p,c,b,PC,CC,PCP,CCP,PCC,PCN,PCCP,PCNP,BC,BCP,BN,BNP,IN])
    
    # Parameter Assignments
    k1    = cs.SX.sym("k1")
    k2    = cs.SX.sym("k2")
    k3    = cs.SX.sym("k3")
    k4    = cs.SX.sym("k4")
    k5    = cs.SX.sym("k5")
    k6    = cs.SX.sym("k6")
    k7    = cs.SX.sym("k7")
    k8    = cs.SX.sym("k8")
    KAP   = cs.SX.sym("KAP")
    KAC   = cs.SX.sym("KAC")
    KIB   = cs.SX.sym("KIB")
    kdmb  = cs.SX.sym("kdmb")
    kdmc  = cs.SX.sym("kdmc")
    kdmp  = cs.SX.sym("kdmp")
    kdn   = cs.SX.sym("kdn")
    kdnc  = cs.SX.sym("kdnc")
    Kd    = cs.SX.sym("Kd")
    Kdp   = cs.SX.sym("Kdp")
    Kp    = cs.SX.sym("Kp")
    KmB   = cs.SX.sym("KmB")
    KmC   = cs.SX.sym("KmC")
    KmP   = cs.SX.sym("KmP")
    ksB   = cs.SX.sym("ksB")
    ksC   = cs.SX.sym("ksC")
    ksP   = cs.SX.sym("ksP")
    m     = cs.SX.sym("m")
    n     = cs.SX.sym("n")
    V1B   = cs.SX.sym("V1B")
    V1C   = cs.SX.sym("V1C")
    V1P   = cs.SX.sym("V1P")
    V1PC  = cs.SX.sym("V1PC")
    V2B   = cs.SX.sym("V2B")
    V2C   = cs.SX.sym("V2C")
    V2P   = cs.SX.sym("V2P")
    V2PC  = cs.SX.sym("V2PC")
    V3B   = cs.SX.sym("V3B")
    V3PC  = cs.SX.sym("V3PC")
    V4B   = cs.SX.sym("V4B")
    V4PC  = cs.SX.sym("V4PC")
    vdBC  = cs.SX.sym("vdBC")
    vdBN  = cs.SX.sym("vdBN")
    vdCC  = cs.SX.sym("vdCC")
    vdIN  = cs.SX.sym("vdIN")
    vdPC  = cs.SX.sym("vdPC")
    vdPCC = cs.SX.sym("vdPCC")
    vdPCN = cs.SX.sym("vdPCN")
    vmB   = cs.SX.sym("vmB")
    vmC   = cs.SX.sym("vmC")
    vmP   = cs.SX.sym("vmP")
    vsB   = cs.SX.sym("vsB")
    vsC   = cs.SX.sym("vsC")
    vsP   = cs.SX.sym("vsP")
    
    paramset = cs.vertcat([k1, k2, k3, k4, k5, k6, k7, k8, KAP, KAC, KIB, kdmb, kdmc, kdmp, kdn,\
                    kdnc, Kd, Kdp, Kp, KmB, KmC, KmP, ksB, ksC, ksP, m, n, V1B, V1C, V1P, V1PC, \
                    V2B, V2C, V2P, V2PC, V3B, V3PC, V4B, V4PC, vdBC, vdBN, vdCC, vdIN, vdPC, \
                    vdPCC, vdPCN, vmB, vmC, vmP, vsB, vsC, vsP])
    
    # Time Variable
    t = cs.SX.sym("t")
    
    
    
    
    #===================================================================
    # set up the ode system
    #===================================================================    
    ode = [[]]*EqCount
    #    /*  mRNA of per */
    ode[0] = (vsP*pow(BN,n)/(pow(KAP,n)  +  pow(BN,n))  -  vmP*p/(KmP  +  p)  -  kdmp*p)
    
    #    /*  mRNA of cry */
    ode[1] = (vsC*pow(BN,n)/(pow(KAC,n)  +  pow(BN,n))  -  vmC*c/(KmC  +  c)  -  kdmc*c)
    
    #    /*  mRNA of BMAL1  */
    ode[2] = (vsB*pow(KIB,m)/(pow(KIB,m) +  pow(BN,m))  -  vmB*b/(KmB  +  b)  -  kdmb *b)
    
    #    /*  protein PER cytosol */
    ode[3] = (ksP*p  -  V1P*PC/(Kp  +  PC)  +  V2P*PCP/(Kdp  +  PCP)  +  k4*PCC  -  k3*PC*CC  -  kdn*PC)
    
    #    /*  protein CRY cytosol */
    ode[4] = (ksC*c - V1C*CC/(Kp + CC) + V2C*CCP/(Kdp + CCP) + k4*PCC - k3*PC*CC - kdnc*CC)
    
    #    /*  phosphorylated PER cytosol */
    ode[5] = (V1P*PC/(Kp + PC) - V2P*PCP/(Kdp + PCP) - vdPC*PCP/(Kdp + PCP) - kdn*PCP)
    
    #    /*  phosphorylated CRY cytosol  */
    ode[6] = (V1C*CC/(Kp + CC) - V2C*CCP/(Kdp + CCP) - vdCC*CCP/(Kd + CCP) - kdn*CCP)
    
    #    /*  PER:CRY complex cytosol */
    ode[7] = ( - V1PC*PCC/(Kp + PCC) + V2PC*PCCP/(Kdp + PCCP) - k4*PCC + k3*PC*CC + k2*PCN - k1*PCC - kdn*PCC)
    
    #    /* PER:CRY complex nucleus */
    ode[8] = ( - V3PC*PCN/(Kp + PCN) + V4PC*PCNP/(Kdp + PCNP) - k2*PCN + k1*PCC - k7*BN*PCN + k8*IN - kdn*PCN)
    
    #    /*  phopshorylated [PER:CRY)c cytosol */
    ode[9] = (V1PC*PCC/(Kp + PCC) - V2PC*PCCP/(Kdp + PCCP) - vdPCC*PCCP/(Kd + PCCP) - kdn*PCCP)
    
    #    /*  phosphorylated [PER:CRY)n */
    ode[10] = (V3PC*PCN/(Kp + PCN) - V4PC*PCNP/(Kdp + PCNP) - vdPCN*PCNP/(Kd + PCNP) - kdn*PCNP)
    
    #    /*  protein BMAL1 cytosol  */
    ode[11] = (ksB*b - V1B*BC/(Kp + BC) + V2B*BCP/(Kdp + BCP) - k5*BC + k6*BN - kdn*BC)
    
    #    /* phosphorylated BMAL1 cytosol */
    ode[12] = (V1B*BC/(Kp + BC) - V2B*BCP/(Kdp + BCP) - vdBC*BCP/(Kd + BCP) - kdn*BCP)
    
    #    /*  protein BMAL1 nucleus */
    ode[13] = ( - V3B*BN/(Kp + BN) + V4B*BNP/(Kdp + BNP) + k5*BC - k6*BN - k7*BN*PCN  +  k8*IN - kdn*BN)
    
    #    /*  phosphorylatd BMAL1 nucleus */
    ode[14] = (V3B*BN/(Kp + BN) - V4B*BNP/(Kdp + BNP) - vdBN*BNP/(Kd + BNP) - kdn*BNP)
    
    #    /*  inactive complex between [PER:CRY)n abd [CLOCK:BMAL1)n */
    ode[15] = ( - k8*IN + k7*BN*PCN - vdIN*IN/(Kd + IN) - kdn*IN)
    
    
    
    ode = cs.vertcat(ode)    
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset),
                       cs.daeOut(ode=ode))
    fn.setOption("name","Leloup 16")
    
    return fn