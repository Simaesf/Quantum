#!/usr/bin/python3

# CQBN_4node.py
# Sima Esfandiarpour Borujeni

"""
    * 5-qubit quantum circuit preparation for a given 4 node Bayesian network using C-QBN method to convert a BN to a QBN
    * Example 4-node Bayesian Network in the paper : Bayesian networks as Quantum circuits:An experimental evaluation
    * Bayes Net obtained from Shenoy and Shenoy (2000)
"""
import numpy as np
import math
from qiskit import BasicAer, IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.compiler import transpile, assemble
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.transpiler import passes, CouplingMap, Layout, PassManager
from qiskit.converters import circuit_to_dag
from qiskit.tools.monitor import job_monitor

##########################################################################################
## We have a five qubit system
## Maximum number of parent nodes is 2
## We need a CCR gate
## Therefore, 1 ancilla qubit

q = QuantumRegister(5)
c = ClassicalRegister(4) 
qc = QuantumCircuit(q,c)

## Rotation for marginal probabilities of IR and OI
## Marginal of IR = (0.75, 0.25)
## Marginal of OI = (0.6, 0.4)

theta_IR = math.atan(np.sqrt(0.25)/np.sqrt(0.75))
theta_OI = math.atan(np.sqrt(0.4)/np.sqrt(0.6))

## Rotation for conditional probability of SM
## When IR = 0, Pr(SM) = (0.3, 0.7)
## When IR = 1, Pr(SM) = (0.8, 0.2)

theta_SM_0 = math.atan(np.sqrt(0.7)/np.sqrt(0.3))
theta_SM_1 = math.atan(np.sqrt(0.2)/np.sqrt(0.8))

## Rotation for conditional probability of SP
## Pr(SP|SM,OI) = (0.9,0.1)  -- (00)
##              = (0.5, 0.5) --  (01)
##              = (0.4, 0.6) -- (10)
##              = (0.2, 0.8) -- (11)
##   last one   = (prob SP low, prob SP high)-- given parents(SM good, OI good)

theta_SP_00 =  math.atan(np.sqrt(0.1)/np.sqrt(0.9))
theta_SP_01 =  math.atan(np.sqrt(0.5)/np.sqrt(0.5))
theta_SP_10 =  math.atan(np.sqrt(0.6)/np.sqrt(0.4))
theta_SP_11 =  math.atan(np.sqrt(0.8)/np.sqrt(0.2))

##  0---1---2
##  1---3---4

## Mapping qubits to Bayesian network nodes depending on the connections
## q[0] --> SP
## q[1] --> Ancilla
## q[3] --> SM
## q[2] --> OI
## q[4] --> IR

## Marginal of IR and OI
qc.ry(2*theta_OI, q[2])
qc.ry(2*theta_IR, q[4])

## Conditional probability of SM (Parent node is IR)
qc.cry(2*theta_SM_1, q[4], q[3]) ## When IR=1
qc.x(q[4])
qc.cry(2*theta_SM_0, q[4], q[3])
qc.x(q[4])

## Conditional rotation of SP with 11 parent states
qc.ccx(q[3], q[2], q[1])
qc.cry(2*theta_SP_11, q[1], q[0])
qc.ccx(q[3], q[2], q[1])

## Conditional rotation of 00
qc.x(q[3])
qc.x(q[2])
qc.ccx(q[3], q[2], q[1])
qc.cry(2*theta_SP_00, q[1], q[0])
qc.ccx(q[3], q[2], q[1])
qc.x(q[2])
qc.x(q[3])

## Conditional rotation of 10 (SM=1, OI=0)
qc.x(q[2])
qc.ccx(q[3], q[2], q[1])
qc.cry(2*theta_SP_10, q[1], q[0])
qc.ccx(q[3], q[2], q[1])
qc.x(q[2])

## Conditional rotation of 01
qc.x(q[3])
qc.ccx(q[3], q[2], q[1])
qc.cry(2*theta_SP_01, q[1], q[0])
qc.ccx(q[3], q[2], q[1])
qc.x(q[3])

### Measuring the state of the qubits:
qc.measure(q[0], c[0]) ### SP --> c[0]
qc.measure(q[2], c[2]) ### OI --> c[2]
qc.measure(q[3], c[1]) ### SM --> c[1]
qc.measure(q[4], c[3]) ### IR --> c[3]


print("Here is the circuit:")
qc.draw(output='mpl')

### Execute circuit on a simulator or computer

IBMQ.load_account()
provider0 = IBMQ.get_provider()
available_backends = provider0.backends() ### All available backends 

print('Available Backends:\n')
for b in available_backends: 
    print(b)

### Choosing the desired backend with at least 5 qubits:

# backend = provider0.get_backend('ibmq_16_melbourne')
# backend = BasicAer.get_backend('qasm_simulator')
# backend = provider0.get_backend('ibmq_qasm_simulator')
backend = provider0.get_backend('ibmq_ourense')
# backend = provider0.get_backend('ibmqx2')
# backend = provider0.get_backend('ibmq_bogota')

### Initiating
marginals = []    
nshots = 8192
allmarginals = []


### Running the circuit 10 times on the backend to perform statistical analysis on the 10 sets of results:

for runnumber in range(10):  
    ### Obtaining the transpiled circuit properties using tdag:  
    transpiled = transpile(qc, backend=backend)
    tdag = circuit_to_dag(transpiled)

    print("backend is:", backend)
    print("the transpiled circuit size is:", transpiled.size())
    print("the transpiled circuit depth is:", transpiled.depth())
    print("The Transpiled circuit properties are:", tdag.properties())

    ### Executing the code with the given number of shots on the device at optimization level 3

    job=execute(qc, backend, shots=nshots, optimization_level=3)
    print("the job id is:", job.job_id())
    job_monitor(job)

    result = job.result()
    print(result.get_counts(qc))

    counts = result.get_counts(qc)
    plot_histogram(counts,  figsize=(12,5) ,bar_labels=True )
   
    ## Postprocessing
    ## Marginal probabilities of the QBN
    probability=[]
    for i in counts.values():
        probability.append(i/nshots)
        
    print ('probabilities of the quantum states are:')
    print("================================================================")
    print(probability)
    print("=================================================================")

    dict_marginal={}   
    node_keys = ['Pr(IR=0)', 'Pr(OI=0)', 'Pr(SM=0)', 'Pr(SP=0)']
    
    for i in range(4):
        counts_i = 0 
        required_states = []
        
        for j in list(counts.keys()): ### keys are the states
            if j[i]=='0':
                required_states.append(j)
                
        for k in required_states:
            counts_i = counts_i + counts[k]
        dict_marginal[node_keys[i]] = counts_i/nshots
        marginals.append(counts_i/nshots)
    print(dict_marginal)
    print("*********************************************************************************")
marg=np.array(marginals)
marg=marg.reshape(10,4)
 

print("Here is the list of marginals for statistical analysis") 
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$") 
print(marg) 
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$") 


marginal_means=[]
marginal_std=[]
for c in range(4):
    aver=np.mean(marg[:,c])
    marginal_means.append(aver)
    stan=np.std(marg[:,c])
    marginal_std.append(stan)
    plt.boxplot(marg[:,c],data=marg)
    keys=np.array(list(dict_marginal.keys()))
print('the summary of the results of 10 runs:')
print(np.array([keys,marginal_means,marginal_std]))
