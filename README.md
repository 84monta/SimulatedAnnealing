# Simulated Annealing for QNNCloud

To compare quantum annealing with traditional way, I created this example code.

# Rquirement

- Python 3.x

***Optional***

if you want to use GPU(CUDA), get librarys as followings:

```
pip install cupy
pip install chainer
```

https://qnncloud.com/index-jp.html

# Usage

no argument. Non :(

Change code directly
 
# Variables



## Simulated Annealing definition
```
#Definition 
T_MAX  = 100.0
T_MIN  = 0.001
MAX_STEMPS = 20

T_LOOP = 600
```

- T_MAX    A starting temperature for a simulation
- T_MIN    A final temperature for a simulation
- MAX_STEMPS How many numbers to chop between T_MAX and T_MIN
- T_LOOP   How many time to try changing status on each temperature

## Input data
You can use data which is donloaded from QNNCloud web page.
```
#f = open('201712030215556824-Jij.json', 'r') # for 100 CHANGE PERSON =100
f = open('201711282214561557-Jij.json', 'r') # for 1000 CHANGE PERSON =1000
#f = open('201712052235195895-Jij.json', 'r') # for 2000 CHANGE PERSON =2000
```

## Size of problem
Ajust "PERSON" to fit problem size.

```
PERSONS = 1000
```

- PERSONS You can change 

## Trial number

```
TRIAL=1
```

- TRIAL How may times do you wanna repeat Simulated Annealing?
