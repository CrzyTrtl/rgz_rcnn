#A script to run demo on a batch of images read from a text file. 
#
#Vinay Kerai (22465472@student.uwa.edu.au)

import os
import csv
import xml.etree.ElementTree as ET
import pandas as pd 

#Name of text file that contains image names
subset = "example.txt"

#Set this to be the path to rgz_rcnn folder
path_U = os.getcwd()

with open(os.path.join(path_U, subset)) as file:
    
    count = 0
    num_to_do = sum(1 for line in file if line.rstrip())
    file.seek(0) 

    for line in file: 

        if count == num_to_do: break

        img = line.rstrip()

        os.chdir(path_U)

        os.system("python demo.py --radio ../data/FITSImages/%s.fits --ir ../data/rgzdemo/%s_infrared.png --plot" % (img,img))
        
        count += 1
        print("|DONE: | " + str(count) + '/' + str(num_to_do))
