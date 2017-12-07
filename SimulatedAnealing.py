import numpy as np
import json
import math
import time

#Definition 
T_MAX  = 800.0
T_MIN  = 1.0
MAX_STEMPS = 100

T_LOOP = 300
PERSONS = 100

#Load Edge Data into narray
#Change file name
f = open('201712030215556824-Jij.json', 'r')
jsonData = json.load(f)
edge_array = np.array(jsonData)

#Create temporary 
person = np.random.choice(2, PERSONS)*2-1
tri_array = np.tri(PERSONS) - np.eye(PERSONS)

## calculate energy
def ret_energy(Edge,Person):
    return np.sum(np.dot(Edge,(tri_array - np.outer(Person, Person) * tri_array)/2.0))

## return move new status or not
def ret_prob(PreE,CurE,temp):
    if PreE < CurE :
        if math.exp((PreE-CurE)/temp) < np.random.rand():
            return False
    return True

#Initialization
steps = 1
current_temp = T_MAX
current_energy = ret_energy(edge_array,person)

#Create new status area
tmp_person = person.copy()

start_time = time.time()
#main loop
while steps < MAX_STEMPS:
    steps = steps + 1
    
    #print current temperature and energy
    print("current temperature = %4.4f , current energy = %f" % (current_temp,current_energy))
    
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
print("**Final temperature = %4.4f , current energy = %f**" % (current_temp,current_energy))
