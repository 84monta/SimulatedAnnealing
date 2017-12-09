import numpy as np
import json
import math
import time

import cupy as cp
import chainer.cuda

#Definition 
T_MAX  = 100.0
T_MIN  = 0.001
MAX_STEMPS = 20

T_LOOP = 600
PERSONS = 1000

GPU = True

TRIAL=1


#Load Edge Data into narray
#Change file name
#f = open('201712030215556824-Jij.json', 'r') # for 100 CHANGE PERSON =100
f = open('201711282214561557-Jij.json', 'r') # for 1000 CHANGE PERSON =1000
#f = open('201712052235195895-Jij.json', 'r') # for 2000 CHANGE PERSON =2000

jsonData = json.load(f)
if GPU :
    edge_array = chainer.cuda.to_gpu(np.array(jsonData))
    tri_array = chainer.cuda.to_gpu(np.tri(PERSONS) - np.eye(PERSONS))
else:
    edge_array = np.array(jsonData)
    tri_array = np.tri(PERSONS) - np.eye(PERSONS)



## calculate energy
def ret_energy(Edge,Person):
    if GPU :
        return cp.sum(Edge * tri_array * (1.0 - cp.outer(Person, Person))/2.0)
    else:
        return np.sum(Edge * tri_array * (1.0 - np.outer(Person, Person))/2.0)
    
## return move new status or not
def ret_prob(PreE,CurE,temp):
    if GPU :
        if PreE < CurE :
            if math.exp((PreE-CurE)/temp) < cp.random.rand():
                return False
        return True
    else:
        if PreE < CurE :
            if math.exp((PreE-CurE)/temp) < np.random.rand():
                return False
        return True
    
for i in range(TRIAL):

    cp.random.seed(i)
    
    #Initialization
    steps = 1
    current_temp = T_MAX
    if GPU :
        person = cp.random.choice(2, PERSONS)*2-1
    else:
        person = np.random.choice(2, PERSONS)*2-1
        
    current_energy = ret_energy(edge_array,person)
    
    
    #Create new status area
    tmp_person = person.copy()
    start_time = time.time()
    #main loop
    while steps < MAX_STEMPS:
        steps = steps + 1
        
        #print current temperature and energy
        print("current temperature = %6.6f , current energy = %f" % (current_temp,current_energy))
        
        #iteration on same temperature
        i = 1
        while i < T_LOOP:
            i = i + 1
            selecter = np.random.randint(PERSONS)
            tmp_person[selecter] = -tmp_person[selecter] 
            tmp_energy = ret_energy(edge_array,tmp_person)
            if ret_prob(current_energy,tmp_energy,current_temp):
                current_energy = tmp_energy
                person[selecter] = -person[selecter]
            else:
                tmp_person[selecter] = -tmp_person[selecter] 
        current_temp = T_MAX * math.exp(-math.log(T_MAX / T_MIN)* steps/MAX_STEMPS)
      
    end_time = time.time() 
     
    print("elapsed_time:{0}".format(end_time - start_time) + "[sec]")
    print("**Final temperature = %6.6f , current energy = %f**" % (current_temp,current_energy))

