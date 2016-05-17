# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:35:45 2014

@author: john
"""

import casadi as cs
import numpy as np

# Sensitivities
abstol = 1e-11
reltol = 1e-9
maxnumsteps = 40000

# Constants and Equation Setup
EqCount = 12
ParamCount = 51

param = [0.3687, 2.2472, 0.2189, 0.5975, 0.7165, 0.5975*1.1444,
            1.0087, 0.9754, 0.9135, 1.9796, 1.4992, 0.6323, 0.0151,
            0.2447, 1.9004, 0.0603, 0.7254, 3.2261, 1.7679, 1.0821,
            0.0931, 1.1853, 0.8856, 0.3118, 0.0236, 1.2620, 1.6294,
            1.0136, 1.8596, 0.7780, 1.9296, 0.7348, 0.1052, 1.4786,
            0.0110, 9.3227, 1.3309, 0.4586, 1.4281, 6.9448, 0.9559,
            1.7986, 0.0151*1.1624, 1.1596, 0.6551, 0.0164, 0.9501,
            0.6327, 0.0954, 0.1338, 0.1907]
            
y0in = [3.18905639,   3.8487871,   2.49418162,  13.5446698,
        0.29856141,   0.09180213,  12.62474256,   4.05666345,
        0.73696792,   0.46171796,   0.58005922,   0.44990306]
        
period = 7.32652564

def model():

    #==================================================================
    #setup of symbolics
    #==================================================================
    MP   = cs.SX.sym("MP")   # mRNA of per
    MC   = cs.SX.sym("MC")   # mRNA of cry
    PC   = cs.SX.sym("PC")   # PER protein (cytosol)
    CC   = cs.SX.sym("CC")   # CRY protein (cytosol)
    PCC  = cs.SX.sym("PCC")  # PER:CRY complex (cytosol)
    PCN  = cs.SX.sym("PCN")  # PER:CRY complex (nucleus)
    BC   = cs.SX.sym("BC")   # protein BMAL1 (cytosol)
    BN   = cs.SX.sym("BN")   # [CLOCK:BMAL1]==nuclear BMAL1 protein
    BNac = cs.SX.sym("BNac") # Transcriptionally active (acetylated) BMAL1 complex
    MN   = cs.SX.sym("MN")   # mRNA of Nampt
    N    = cs.SX.sym("N")    # protein NAMPT
    NAD  = cs.SX.sym("NAD")  # cellular NAD levels
    
    sys = cs.vertcat([MP, MC, PC, CC, PCC, PCN, BC, BN, BNac, MN, N, NAD])

    #===================================================================
    #Parameter definitions
    #===================================================================
    n_n = 1; 
    n_p = 1; 
    n_c = 1; 
    r_n = 3; 
    r_p = 3; 
    r_c = 3; 
    
    V1PC     = cs.SX.sym("V1PC")
    KAN      = cs.SX.sym("KAN")
    V3PC     = cs.SX.sym("V3PC")
    k1       = cs.SX.sym("k1")
    vmN      = cs.SX.sym("vmN")
    k2       = cs.SX.sym("k2")
    Kp_pcc   = cs.SX.sym("Kp_pcc")
    Kp_bc    = cs.SX.sym("Kp_bc")
    KmN      = cs.SX.sym("KmN")
    Kp_c     = cs.SX.sym("Kp_c")
    vmP      = cs.SX.sym("vmP")
    k5       = cs.SX.sym("k5")
    k3       = cs.SX.sym("k3")
    V1B      = cs.SX.sym("V1B")
    vsn      = cs.SX.sym("vsn")
    V1C      = cs.SX.sym("V1C")
    Kp_pcn   = cs.SX.sym("Kp_pcn")
    Kac_bn   = cs.SX.sym("Kac_bn")
    Kd_bn    = cs.SX.sym("Kd_bn")
    V4B      = cs.SX.sym("V4B")
    Kdac_bn  = cs.SX.sym("Kdac_bn")
    MB0      = cs.SX.sym("MB0")
    KAP      = cs.SX.sym("KAP")
    KAC      = cs.SX.sym("KAC")
    vdBN     = cs.SX.sym("vdBN")
    V3B      = cs.SX.sym("V3B")
    ksN      = cs.SX.sym("ksN")
    sn       = cs.SX.sym("sn")
    vm_NAMPT = cs.SX.sym("vm_NAMPT")
    vm_NAD   = cs.SX.sym("vm_NAD")
    Km_NAMPT = cs.SX.sym("Km_NAMPT")
    Km_NAD   = cs.SX.sym("Km_NAD")
    v0p      = cs.SX.sym("v0p")
    v0c      = cs.SX.sym("v0c")
    v0n      = cs.SX.sym("v0n")
    vsP      = cs.SX.sym("vsP")
    vsC      = cs.SX.sym("vsC")
    kd       = cs.SX.sym("kd")
    k6       = cs.SX.sym("k6")
    ksB      = cs.SX.sym("ksB")
    ksP      = cs.SX.sym("ksP")
    ksC      = cs.SX.sym("ksC")
    k4       = cs.SX.sym("k4")
    Kp_p     = cs.SX.sym("Kp_p")
    vmC      = cs.SX.sym("vmC")
    KmP      = cs.SX.sym("KmP")
    KmC      = cs.SX.sym("KmC")
    V1P      = cs.SX.sym("V1P")
    Rp       = cs.SX.sym("Rp")
    Rc       = cs.SX.sym("Rc")
    Rn       = cs.SX.sym("Rn")

    paramset = cs.vertcat(
        [V1PC, KAN, V3PC, k1, vmN, k2, Kp_pcc, Kp_bc, KmN, Kp_c, vmP,
         k5, k3, V1B, vsn, V1C, Kp_pcn, Kac_bn, Kd_bn, V4B, Kdac_bn,
         MB0, KAP, KAC, vdBN, V3B, ksN, sn, vm_NAMPT, vm_NAD, Km_NAMPT,
         Km_NAD, v0p, v0c, v0n, vsP, vsC, kd, k6, ksB, ksP, ksC, k4,
         Kp_p, vmC, KmP, KmC, V1P, Rp, Rc, Rn])

    #time
    t = cs.SX.sym("t")
    
    ode = [[]]*EqCount #initializes vector
    
    ode[0]  = v0p + vsP*BNac**n_p/(KAP**n_p*(1 + (PCN/Rp)**r_p) + BNac**n_p)-vmP*MP/(KmP+MP)-kd*MP       # Per
    ode[1]  = v0c + vsC*BNac**n_c/(KAC**n_c*(1 + (PCN/Rc)**r_c) + BNac**n_c) - vmC*MC/(KmC+MC)- kd*MC    # Cry
    ode[2]  = ksP*MP + k4*PCC - k3*PC*CC - V1P*PC/(Kp_p + PC) - kd*PC                                    # PER CYTOSOL
    ode[3]  = ksC*MC + k4*PCC - k3*PC*CC - V1C*CC/(Kp_c + CC) - kd*CC                                    # CRY CYTOSOL
    ode[4]  = (k3*PC*CC + k2*PCN - k4*PCC - k1*PCC - V1PC*PCC/(Kp_pcc + PCC) - kd*PCC)                   # PER/CRY CYTOSOL COMPLEX
    ode[5]  = k1*PCC - k2*PCN  - V3PC*PCN/(Kp_pcn + PCN) - kd*PCN                                        # PER/CRY NUCLEUS
    ode[6]  = ksB*MB0 - k5*BC + k6*BN - V1B*BC/(Kp_bc + BC) - kd*BC                                      # BMAL CYTOSOL
    ode[7]  = (k5*BC-k6*BN-V3B*BN/(Kac_bn+BN) + V4B*NAD*BNac/(Kdac_bn+BNac)-vdBN*BN/(Kd_bn+BN) -kd*BN)   # BMAL1 NUCLEUS
    ode[8]  = V3B*BN/(Kac_bn+BN) -V4B*NAD*BNac/(Kdac_bn+BNac)-kd*BNac                                    # BMAL1_ACETYL
    ode[9]  = (v0n+vsn*BNac**n_n/(KAN**n_n*(1+(PCN/Rn)**r_n)+BNac**n_n) - vmN*MN/(KmN+MN) - kd*MN)       # Nampt
    ode[10] = ksN*MN - vm_NAMPT*N/(Km_NAMPT+N) - kd*N                                                    # NAMPT PROTEIN
    ode[11] = sn*N - vm_NAD*NAD/(Km_NAD+NAD) - kd*NAD                                                    # NAD+ cellular levels (~deacetylase activity)

    ode     = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset),
                       cs.daeOut(ode=ode))
    fn.setOption("name","Pegy's Model")
    
    return fn
