import os
from os import listdir
from os.path import isfile, join
mypath="hands"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
signs=set()

for s in onlyfiles:
    elements = s.split('_')
    if str(elements[0]) not in signs: # male
        signs.add(str(elements[0]))
        new_name= mypath+'/'+str(elements[0])
        os.mkdir(new_name)
        os.rename(mypath+'/'+s, mypath+'/'+str(elements[0])+'/'+s)
    else:
        os.rename(mypath+'/'+s, mypath+'/'+str(elements[0])+'/'+s)



