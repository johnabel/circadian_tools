"""
Model from Vasalou Herzog Henzon 2009

Uses only two cells here, since it is the most basic example.

created 3 June 2015, jha
"""

# common imports
from __future__ import division

# python packages
import numpy as np
import casadi as cs
import gillespy as gl
import matplotlib.pyplot as plt

modelversion = 'kiss_oscillator_2011'

# constants and equations setup
neq = 21*2
npa  = 81
param = [0.45, 0.1, 0.4, 0.2, 0.4, 0.2, 0.5, 0.1, 0.6, 0.6, 2.2, 0.01, 0.01,\ 
         0.01, 0.12, 0.01, 0.3, 0.1, 0.1, 0.4, 0.4, 0.31, 0.12, 1.6, 0.6, 4,\ 
         2, 0.4, 0.5, 0.6, 0.4, 0.4, 0.1, 0.1, 0.3, 0.1, 0.5, 0.4, 0.2, 0.1,\ 
         0.5, 0.6, 0.7, 0.8, 0.7, 0.7, 0.7, 0.8, 1.0, 1.1, 1, 1.1, 1.5, 1,\ 
         0.06, 10, 0.5, 5, 1, 5, 2.5, 0.01, 0.01, 1, 1.5, 0.15, 1, 5, 0.001,\ 
         0.5, 400, 6, 3, 0.67, 4.2, 149.5, 5, 2.2, 4.2, 2.1, 1.8]
         
y0in = [-1.29374616,  0.67413515, -1.62914683,  0.87060893]
period = 24


def vasalou_model_ode():
    
    # set up symbolics
    k1 = cs.ssym('k1')
    k2 = cs.ssym('k2')
    k3 = cs.ssym('k3')
    k4 = cs.ssym('k4')
    k5 = cs.ssym('k5')
    k6 = cs.ssym('k6')
    k7 = cs.ssym('k7')
    k8 = cs.ssym('k8')
    KAP = cs.ssym('KAP')
    KAC = cs.ssym('KAC')
    KIB = cs.ssym('KIB')
    kdmb = cs.ssym('kdmb')
    kdmc = cs.ssym('kdmc')
    kdmp = cs.ssym('kdmp')
    kdnc = cs.ssym('kdnc')
    kdn = cs.ssym('kdn')
    Kd = cs.ssym('Kd')
    Kdp = cs.ssym('Kdp')
    Kp = cs.ssym('Kp')
    KmB = cs.ssym('KmB')
    KmC = cs.ssym('KmC')
    KmP = cs.ssym('KmP')
    ksB = cs.ssym('ksB')
    ksC = cs.ssym('ksC')
    ksP = cs.ssym('ksP')
    n = cs.ssym('n')
    m = cs.ssym('m')
    Vphos = cs.ssym('Vphos')
    V1P = cs.ssym('V1P')# Vphos
    V1PC = cs.ssym('V1PC')# Vphos
    V3PC = cs.ssym('V3PC')# Vphos
    V1B = cs.ssym('V1B')
    V1C = cs.ssym('V1C')
    V2B = cs.ssym('V2B')
    V2C = cs.ssym('V2C')
    V2P = cs.ssym('V2P')
    V2PC = cs.ssym('V2PC')
    V3B = cs.ssym('V3B')
    V4B = cs.ssym('V4B')
    V4PC = cs.ssym('V4PC')
    vdBC = cs.ssym('vdBC')
    vdBN = cs.ssym('vdBN')
    vdCC = cs.ssym('vdCC')
    vdIN = cs.ssym('vdIN')
    vdPC = cs.ssym('vdPC')
    vdPCC = cs.ssym('vdPCC')
    vdPCN = cs.ssym('vdPCN')
    vmB= cs.ssym('vmB') # not used
    vmC = cs.ssym('vmC')
    vmP = cs.ssym('vmP')
    vsB = cs.ssym('vsB')#1; not used
    vsC = cs.ssym('vsC')
    KD = cs.ssym('KD')#2;%*scale;
    vP = cs.ssym('vP')
    VMK=cs.ssym('VMK')
    K_1 = cs.ssym('K_1')
    K_2 = cs.ssym('K_2')
    WT = cs.ssym('WT')
    CT = cs.ssym('CT')
    KC = cs.ssym('KC')
    vsP0 = cs.ssym('vsP0')#%1.1; not used
    kf=cs.ssym('kf')
    IP3=cs.ssym('IP3')
    VM3= cs.ssym('VM3')
    M3= cs.ssym('M3')
    KR= cs.ssym('KR')
    KA = cs.ssym('KA')
    pA = cs.ssym('pA')
    VM2=cs.ssym('VM2')
    K2=cs.ssym('K2')
    M2= cs.ssym('M2')
    kMK=cs.ssym('kMK')
    V_b=cs.ssym('V_b')
    k_b=cs.ssym('k_b')
    
    paramset = cs.vertcat([k1, k2, k3, k4, k5, k6, k7, k8, KAP, KAC, KIB, 
                           kdmb, kdmc, kdmp, kdnc, kdn, Kd, Kdp, Kp, KmB, 
                           KmC, KmP, ksB, ksC, ksP, n, m, Vphos, V1B, V1C, 
                           V1P, V1PC, V2B, V2C, V2P, V2PC, V3B, V3PC, V4B, 
                           V4PC, vdBC, vdBN, vdCC, vdIN, vdPC, vdPCC, vdPCN, 
                           vmB, vmC, vmP, vsB, vsC, vsP, RT, KD, k, v0, v1, 
                           vP, VMK, Ka, K_1, K_2, WT, CT, KC, vsP0, v2, kf, 
                           IP3, VM3, M3, KR, KA, pA, VM2, K2, M2, kMK, V_b, 
                           k_b])
    
    # model components
    Ca_1 = cs.ssym('Ca_1')
    Ca_store_1 =   cs.ssym('Ca_store_1')
    MP_1 =  cs.ssym('MP_1')
    MC_1 =  cs.ssym('MC_1')
    MB_1 =  cs.ssym('MB_1')
    PC_1 =  cs.ssym('PC_1')
    CC_1 =  cs.ssym('CC_1')
    PCP_1 =  cs.ssym('PCP_1')
    CCP_1 =  cs.ssym('CCP_1')
    PCC_1 =  cs.ssym('PCC_1')
    PCN_1 =  cs.ssym('PCN_1')
    PCCP_1 =  cs.ssym('PCCP_1')
    PCNP_1 =  cs.ssym('PCNP_1')
    BC_1 =   cs.ssym('BC_1')
    BCP_1 =  cs.ssym('BCP_1')
    BN_1 =  cs.ssym('BN_1')
    BNP_1 =  cs.ssym('BNP_1')
    IN_1 =  cs.ssym('IN_1')
    CB_1 =  cs.ssym('CB_1')
    vVIP_1 =  cs.ssym('vVIP_1')
    gGABA_1 = cs.ssym('gGABA_1')
    
    Ca_2 = cs.ssym('Ca_2')
    Ca_store_2 =   cs.ssym('Ca_store_2')
    MP_2 =  cs.ssym('MP_2')
    MC_2 =  cs.ssym('MC_2')
    MB_2 =  cs.ssym('MB_2')
    PC_2 =  cs.ssym('PC_2')
    CC_2 =  cs.ssym('CC_2')
    PCP_2 =  cs.ssym('PCP_2')
    CCP_2 =  cs.ssym('CCP_2')
    PCC_2 =  cs.ssym('PCC_2')
    PCN_2 =  cs.ssym('PCN_2')
    PCCP_2 =  cs.ssym('PCCP_2')
    PCNP_2 =  cs.ssym('PCNP_2')
    BC_2 =   cs.ssym('BC_2')
    BCP_2 =  cs.ssym('BCP_2')
    BN_2 =  cs.ssym('BN_2')
    BNP_2 =  cs.ssym('BNP_2')
    IN_2 =  cs.ssym('IN_2')
    CB_2 =  cs.ssym('CB_2')
    vVIP_2 =  cs.ssym('vVIP_2')
    gGABA_2 = cs.ssym('gGABA_2')
    
    y = cs.vertcat([Ca_1, Ca_store_1, MP_1, MC_1, MB_1, PC_1, CC_1, PCP_1, 
                    CCP_1, PCC_1, PCN_1, PCCP_1, PCNP_1, BC_1, BCP_1, BN_1, 
                    BNP_1, IN_1, CB_1, vVIP_1, gGABA_1,
                    #
                    Ca_2, Ca_store_2, MP_2, MC_2, MB_2, PC_2, CC_2, PCP_2, 
                    CCP_2, PCC_2, PCN_2, PCCP_2, PCNP_2, BC_2, BCP_2, BN_2, 
                    BNP_2, IN_2, CB_2, vVIP_2, gGABA_2,])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    # ode system