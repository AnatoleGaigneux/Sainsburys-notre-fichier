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

#initialize arrays
Ele_cat1, Ele_cat2, Ele_cat3 = ([] for i in range(3))
Gas_cat1, Gas_cat2, Gas_cat3 = ([] for j in range(3))
Carbon_cat1, Carbon_cat2, Carbon_cat3 = ([] for k in range(3))
Ele_carbon_cat1, Ele_carbon_cat2, Ele_carbon_cat3 = ([] for l in range(3))
Gas_carbon_cat1, Gas_carbon_cat2, Gas_carbon_cat3 = ([] for m in range(3))
category = []

for store_index in range(id_store_min, id_store_max):

    cur.execute('''SELECT Age, Area, Carbon, ED2016, GD2016, Elec_Emissions, Gas_Emissions FROM Stores Where id= {vn1}'''.format(vn1=store_index))
    Index = cur.fetchall()
    if not Index:
        pass
    else:
        Age = np.array([elt[0] for elt in Index])
        Area = np.array([elt[1] for elt in Index])
        Carbon = np.array([elt[2] for elt in Index])
        Ele = np.array([elt[3] for elt in Index])
        Gas = np.array([elt[4] for elt in Index])
        Ele_Emissions = np.array([elt[5] for elt in Index])
        Gas_Emissions = np.array([elt[6] for elt in Index])

        if Area <= 25000:
            category = 1

        elif 25000 < Area <= 45000:
            category = 2

        elif Area > 45000:
            category = 3

        else:
            print("Not able to categorise")

        if category == 1:
            Ele_cat1.append(Ele)
            Gas_cat1.append(Gas)
            Carbon_cat1.append(Carbon)
            Ele_carbon_cat1.append(Ele_Emissions)
            Gas_carbon_cat1.append(Gas_Emissions)
        elif category == 2:
            Ele_cat2.append(Ele)
            Gas_cat2.append(Gas)
            Carbon_cat2.append(Carbon)
            Ele_carbon_cat2.append(Ele_Emissions)
            Gas_carbon_cat2.append(Gas_Emissions)
        elif category == 3:
            Ele_cat3.append(Ele)
            Gas_cat3.append(Gas)
            Carbon_cat3.append(Carbon)
            Ele_carbon_cat3.append(Ele_Emissions)
            Gas_carbon_cat3.append(Gas_Emissions)



Gas_carbon_average = [np.average(Gas_carbon_cat1), np.average(Gas_carbon_cat2), np.average(Gas_carbon_cat3)]
Ele_carbon_average = [np.average(Ele_carbon_cat1), np.average(Ele_carbon_cat2), np.average(Ele_carbon_cat3)]
Gas_carbon_std = [np.std(Gas_carbon_cat1), np.std(Gas_carbon_cat2), np.std(Gas_carbon_cat3)]
Ele_carbon_std = [np.std(Ele_carbon_cat1), np.std(Ele_carbon_cat2), np.std(Ele_carbon_cat3)]

# Plot emissions
ind = np.arange(3) # number of bars
width = 0.35
plt.figure(2)
p3 = plt.bar(ind, Ele_carbon_average, width, color='#d62728', yerr=Ele_carbon_std, capsize=4, ecolor='red')
p4 = plt.bar(ind, Gas_carbon_average, width, bottom=Ele_carbon_average, yerr=Gas_carbon_std, capsize=4, ecolor='blue')

plt.ylabel('Carbon emissions 2016-17 (tCO2)')
plt.title('Carbon emissions per category of stores')
plt.xticks(ind, ('Cat 1 #stores:%s' % len(Ele_carbon_cat1), 'Cat 2 #stores:%s' % len(Ele_carbon_cat2), 'Cat 3 #stores:%s' % len(Ele_carbon_cat3)))
plt.legend((p3[0], p4[0]), ('Electricity', 'Gas'))

plt.show()
