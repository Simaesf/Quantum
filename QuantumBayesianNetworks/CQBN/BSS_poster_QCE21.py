#!/usr/bin/python
# coding: utf-8
# Sima Esfandiarpour 
# This code provides an example of building a quantum circuit using C-QBN for a 5 node Bayesian network
# there are 3 layers of variables in this BN :In the following graph of the BN, the edges are directed and are downwards. 
# d takes 4 values, 0,1,2,3 and the rest of variables are binary.

#               d
#             / | \
#            /  |  \
#          TC  MC   PC
#           \   |   /
#            \  |  /
#              OC

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from qiskit import IBMQ , BasicAer, Aer, QuantumCircuit,  ClassicalRegister, QuantumRegister, execute
from qiskit.compiler import transpile
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from numpy import pi
from qiskit.test.mock import FakeMelbourne


## The function for calculating the rotation angle

def angle(pzero,pone):
    theta = 2*math.atan(np.sqrt(pone)/np.sqrt(pzero))
    return theta

# ## Variables and the values they can take, and their CPTs

d = ['{0:02b}'.format(i) for i in range(4)]
# d = [0,1,2,3]
TC= ['{0:01b}'.format(i) for i in range(2)]
# TC = ['0', '1']
MC = ['{0:01b}'.format(i) for i in range(2)]
# MC = ['0', '1']
PC = ['{0:01b}'.format(i) for i in range(2)]
# PC = ['0', '1']
OC = ['{0:01b}'.format(i) for i in range(2)]
# OC = ['0', '1']
nodes = [d, TC, MC, PC, OC]
#----------------------------------------------------------------

## CPTs:
## d --> q[0], q[1]
# d = [prob(0), prob(1), prob(2), prob(3)] or simply 00: for 0, 01: for 1, 10: for 2, 11: for 3

# We are running the code for different value sets for d_probs
d_probs = [0.25, 0.25, 0.25, 0.25]
# d_probs = [0, 1, 0, 0]
#d_probs = [0.00, 0.00, 1.00, 0.00]

## TC --> q[2]
#  0  , 1 given d= ...
TC_probs =  [[0.85, 0.15], #d=00
             [0.82, 0.18], #d=01
             [0.67, 0.33], #d=10
             [0.65, 0.35]] #d=11

## MC --> q[3]
#  0  , 1 given d= ...
MC_probs =  [[0.80, 0.20], #d=00
             [0.60, 0.40], #d=01
             [0.45, 0.55], #d=10
             [0.45, 0.55]] #d=11

## PC --> q[4]
#  0  , 1 given d= ...
PC_probs =  [[0.33, 0.67], #d=00
             [0.67, 0.33], #d=01
             [1.00, 0.00], #d=10
             [1.00, 0.00]] #d=11

## OC --> q[5]
#  0  , 1 given TC, MC, PC= ijk where i,j,k \in {0,1}

OC_probs =  [[0.90, 0.10], #000
             [0.55, 0.45], #001
             [0.45, 0.55], #010
             [0.60, 0.40], #011
             [0.65, 0.35], #100
             [0.35, 0.65], #101
             [0.20, 0.80], #110
             [0.30, 0.70]] #111 

## Calculating all the rotation angles using the angle function

################################################################
# rotation angles for d
# def angle(pzero,pone):
#     theta = 2*math.atan(np.sqrt(pone)/np.sqrt(pzero))
#     return theta

theta_1d= angle(d_probs[0]+d_probs[1], d_probs[2]+d_probs[3])
theta_2d_1d0 = angle(d_probs[0], d_probs[1])
theta_2d_1d1 = angle(d_probs[2], d_probs[3])
################################################################


theta_TC_d00 = angle(TC_probs[0][0], TC_probs[0][1])
theta_TC_d01 = angle(TC_probs[1][0], TC_probs[1][1])
theta_TC_d10 = angle(TC_probs[2][0], TC_probs[2][1])
theta_TC_d11 = angle(TC_probs[3][0], TC_probs[3][1])


theta_MC_d00 = angle(MC_probs[0][0], MC_probs[0][1])
theta_MC_d01 = angle(MC_probs[1][0], MC_probs[1][1])
theta_MC_d10 = angle(MC_probs[2][0], MC_probs[2][1])
theta_MC_d11 = angle(MC_probs[3][0], MC_probs[3][1])

theta_PC_d00 = angle(PC_probs[0][0], PC_probs[0][1])
theta_PC_d01 = angle(PC_probs[1][0], PC_probs[1][1])
theta_PC_d10 = angle(PC_probs[2][0], PC_probs[2][1])
theta_PC_d11 = angle(PC_probs[3][0], PC_probs[3][1])

################################################################
theta_OC_000 = angle(OC_probs[0][0], OC_probs[0][1])
theta_OC_001 = angle(OC_probs[1][0], OC_probs[1][1])
theta_OC_010 = angle(OC_probs[2][0], OC_probs[2][1])
theta_OC_011 = angle(OC_probs[3][0], OC_probs[3][1])
theta_OC_100 = angle(OC_probs[4][0], OC_probs[4][1])
theta_OC_101 = angle(OC_probs[5][0], OC_probs[5][1])
theta_OC_110 = angle(OC_probs[6][0], OC_probs[6][1])
theta_OC_111 = angle(OC_probs[7][0], OC_probs[7][1])


## CONSTRUCTING THE CIRCUIT--------------------------------
## We have a 8 qubit system
## we need a maximum of CCCR gate
## Therefore, 2 ancilla qubit

# ----------------------------------------------------------
# +++++++++++++++++++++++++ C-QBN +++++++++++++++++++++++++
# ----------------------------------------------------------
## q[0], q[1] --> d
## q[2] --> TC
## q[3] --> MC
## q[4] --> PC
## q[5], q[6] --> ancilla
## q[7] --> OC

# ----------------------------------------------------------

q = QuantumRegister(8, 'q')
c = ClassicalRegister( 6, 'c')
qc = QuantumCircuit(q,c)

## Marginal of d 
qc.ry(theta_1d, q[0])
qc.x(q[0])
qc.cry(theta_2d_1d0, q[0], q[1])
qc.x(q[0])
qc.cry(theta_2d_1d1, q[0], q[1])


## Conditionals of  TC
qc.x(q[0])
qc.x(q[1])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_TC_d00, q[5], q[2])
qc.ccx(q[0], q[1],q[5])
qc.x(q[1])
qc.x(q[0])


qc.x(q[0])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_TC_d01, q[5], q[2])
qc.ccx(q[0], q[1],q[5])
qc.x(q[0])


qc.x(q[1])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_TC_d10, q[5], q[2])
qc.ccx(q[0], q[1],q[5])
qc.x(q[1])


qc.ccx(q[0], q[1],q[5])
qc.cry(theta_TC_d11, q[5], q[2])
qc.ccx(q[0], q[1],q[5])



## Conditionals of  MC
qc.x(q[0])
qc.x(q[1])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_MC_d00, q[5], q[3])
qc.ccx(q[0], q[1],q[5])
qc.x(q[1])
qc.x(q[0])


qc.x(q[0])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_MC_d01, q[5], q[3])
qc.ccx(q[0], q[1],q[5])
qc.x(q[0])


qc.x(q[1])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_MC_d10, q[5], q[3])
qc.ccx(q[0], q[1],q[5])
qc.x(q[1])


qc.ccx(q[0], q[1],q[5])
qc.cry(theta_MC_d11, q[5], q[3])
qc.ccx(q[0], q[1],q[5])


## Conditionals of  PC
qc.x(q[0])
qc.x(q[1])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_PC_d00, q[5], q[4])
qc.ccx(q[0], q[1],q[5])
qc.x(q[1])
qc.x(q[0])


qc.x(q[0])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_PC_d01, q[5], q[4])
qc.ccx(q[0], q[1],q[5])
qc.x(q[0])


qc.x(q[1])
qc.ccx(q[0], q[1],q[5])
qc.cry(theta_PC_d10, q[5], q[4])
qc.ccx(q[0], q[1],q[5])
qc.x(q[1])


qc.ccx(q[0], q[1],q[5])
qc.cry(theta_PC_d11, q[5], q[4])
qc.ccx(q[0], q[1],q[5])

#######################################

## Conditionals of  OC
# parent node (qubits) q2,q3,q4 are 000
qc.x(q[2])
qc.x(q[3])
qc.x(q[4])
qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_000, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])
qc.x(q[4])
qc.x(q[3])
qc.x(q[2])

######################################

# parent node (qubits) q2,q3,q4 are 001
qc.x(q[2])
qc.x(q[3])

qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_001, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])

qc.x(q[3])
qc.x(q[2])

######################################

# parent node (qubits) q2,q3,q4 are 010
qc.x(q[2])

qc.x(q[4])
qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_010, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])
qc.x(q[4])

qc.x(q[2])

######################################

# parent node (qubits) q2,q3,q4 are 011
qc.x(q[2])

qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_011, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])

qc.x(q[2])

######################################

# parent node (qubits) q2,q3,q4 are 100

qc.x(q[3])
qc.x(q[4])
qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_100, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])
qc.x(q[4])
qc.x(q[3])

######################################

# parent node (qubits) q2,q3,q4 are 101

qc.x(q[3])

qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_101, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])

qc.x(q[3])


######################################

# parent node (qubits) q2,q3,q4 are 110

qc.x(q[4])
qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_110, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])
qc.x(q[4])


######################################

# parent node (qubits) q2,q3,q4 are 111

qc.ccx(q[2], q[3],q[5])
qc.ccx(q[4], q[5],q[6])
qc.cry(theta_OC_111, q[6], q[7])
qc.ccx(q[4], q[5],q[6])
qc.ccx(q[2], q[3],q[5])

qc.measure([0,1,2,3,4,7], [0,1,2,3,4,5])

# Get backend from Aer provider
nshots = 8192
simulator_backend = Aer.get_backend('qasm_simulator')

## Drawing the quantum circuit
circuit.draw()

## Running the circuit on the backend
job = execute(qc, backend=simulator_backend, shots=nshots)
result = job.result()
counts = result.get_counts()

## histogram of joint probabilities
plot_histogram(counts, figsize=(20,5),number_to_keep=35)


## POSTPROCESSING
## CALCULATING Marginal probabilities from the backend results
######################################################################
marginals=[]
probability=[]
for i in counts.values():
    probability.append(i/nshots)

print("================================================================================================================================================")
dict_marginal={}   
node_keys = ['Pr(d=0)', 'Pr(d=1)', 'Pr(d=2)', 'Pr(d=3)', 'Pr(TC=0)', 'Pr(TC=1)', 'Pr(MC=0)', 'Pr(MC=1)', 'Pr(PC=0)', 'Pr(PC=1)', 'Pr(OC=0)', 'Pr(OC=1)']

required_states =[]
required_states_d0 = []
required_states_d1 = []
required_states_d2 = []
required_states_d3 = []
required_states_TC0 = []
required_states_TC1 = []

required_states_MC0 = []
required_states_MC1 = []

required_states_PC0 = []
required_states_PC1 = []

required_states_OC0 = []
required_states_OC1 = []



for j in list(counts.keys()): ## keys are the states
    if j[5]=='0'and j[4]=='0':
        required_states_d0.append(j)
    if j[5]=='0'and j[4]=='1':
        required_states_d1.append(j)
    if j[5]=='1'and j[4]=='0':
        required_states_d2.append(j)
    if j[5]=='1'and j[4]=='1':
        required_states_d3.append(j)
    if j[3]=='0':
        required_states_TC0.append(j)
    if j[3]=='1':
        required_states_TC1.append(j)        
    if j[2]=='0':
        required_states_MC0.append(j)
    if j[2]=='1':
        required_states_MC1.append(j)
    if j[1]=='0':
        required_states_PC0.append(j)
    if j[1]=='1':
        required_states_PC1.append(j)
    if j[0]=='0':
        required_states_OC0.append(j)
    if j[0]=='1':
        required_states_OC1.append(j)
        
required_states.append(required_states_d0) 
required_states.append(required_states_d1)
required_states.append(required_states_d2) 
required_states.append(required_states_d3) 

required_states.append(required_states_TC0)
required_states.append(required_states_TC1)
required_states.append(required_states_MC0)
required_states.append(required_states_MC1)
required_states.append(required_states_PC0)
required_states.append(required_states_PC1)
required_states.append(required_states_OC0)
required_states.append(required_states_OC1)

for k in required_states:
    counts_i = 0 
    for r in k:
        counts_i = counts_i + counts[r]
    dict_marginal[node_keys[required_states.index(k)]] = counts_i/nshots
    marginals.append(counts_i/nshots)
print ("marginal probabilities of the BN on Aer qasm_simulator")
print(dict_marginal)
print("*********************************************************************************")

  


