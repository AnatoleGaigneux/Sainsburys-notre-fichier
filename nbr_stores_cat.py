# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:37:48 2017

@author: Ttle
"""


import sqlite3
#import Common.classStore as st
#import Common.classTech as tc
#import Solvers.classCHPProblem as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

database_path = ".\\Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()


id_store_min = 0
id_store_max = 3000

time_start = 806448
time_stop = 824016

n1=0
n2=0
n3=0

for id_store in range(id_store_min, id_store_max ):
    goodAAC = 0
    cur.execute('''SELECT Area FROM Stores Where id= {vn1}'''.format(vn1=id_store))
    checkAAC = cur.fetchall()
    try:
        if checkAAC[0][0] > 0:
            goodAAC = 1

    except:
        pass

    if goodAAC == 1:
        cur.execute(
            '''SELECT Area FROM Stores Where id= {vn1}'''.format(
                vn1=id_store))
        Index = cur.fetchall()
        if not Index:
            pass
        else:
            Area = np.array([elt[0] for elt in Index])

            if Area <= 25000:
                category = 1

            elif 25000 < Area <= 45000:
                category = 2

            elif Area > 45000:
                category = 3

            else:
                print("Not able to categorise")
            if category == 1:
                n1= n1 + 1
            if category == 2:
                n2 = n2 + 1
            if category == 3:
                n3 = n3 + 1

print(n1,n2,n3)
