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
id_store_max = 2300

time_start = 806448
time_stop = 824016

#initialize arrays
Ele_cat1, Ele_cat2, Ele_cat3, Ele_cat4, Ele_cat5, Ele_cat6 = ([] for i in range(6))
Gas_cat1, Gas_cat2, Gas_cat3, Gas_cat4, Gas_cat5, Gas_cat6 = ([] for j in range(6))
Carbon_cat1, Carbon_cat2, Carbon_cat3, Carbon_cat4, Carbon_cat5, Carbon_cat6 = ([] for k in range(6))

Age = []
Area = []
Ele_annual_demand = []
Gas_annual_demand = []
Ref_annual_demand = []
Carbon_all = []

for store_index in range(id_store_min, id_store_max ):
    goodAAC = 0
    cur.execute('''SELECT Age, Area, Carbon, ED2016 FROM Stores Where id= {vn1}'''.format(vn1=store_index))
    checkAAC = cur.fetchall()
    try:
        if checkAAC[0][0] > 0:
            if checkAAC[0][1] > 0:
                if checkAAC[0][2] > 0:
                    if checkAAC[0][3] > 0:
                        goodAAC = 1
    except:
        pass

    if goodAAC == 1:
        cur.execute('''SELECT AGE, AREA, CARBON, ED2016, GD2016 FROM Stores Where id= {vn1}'''.format(vn1=store_index))
        Index = cur.fetchall()
        try:
            Age.append(Index[0][0])
            Area.append(Index[0][1])
            Carbon_all.append(Index[0][2])
            Ele_annual_demand.append(Index[0][3])
            Gas_annual_demand.append(Index[0][4])
                #print(store_index
        except:
            pass

print(len(Area), len(Age), len(Ele_annual_demand))

fig = plt.figure(2) # 3D plot of electricity and gas demand vs Area and Age
ax = Axes3D(fig)
ax.scatter(Area, Age, Ele_annual_demand, zdir='z', s=20, c='r', depthshade=True, label='Electricity')
ax.set_xlabel('Area (ft2)')
ax.set_ylabel('Age (yr)')
ax.set_zlabel('Annual energy demand (kW)')
ax.tick_params(axis='both', which='major', pad=-5)
plt.show()
