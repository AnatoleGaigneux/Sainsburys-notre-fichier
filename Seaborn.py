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
from operator import add
import seaborn as sns
import csv

database_path = ".\\Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()


id_store_min = 1
id_store_max = 3000

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
Location = []

for store_index in range(id_store_min, id_store_max ):
    goodIO = 0
    cur.execute('''SELECT Age, Area, Carbon, ED2016, GD2016, Zone FROM Stores Where id= {vn1}'''.format(vn1=store_index))
    data = cur.fetchall()
    try:
        if data[0][0] > 0:
            if data[0][1] > 0:
                if data[0][2] > 0:
                    if data[0][3] > 0:
                        if data[0][4] > 0:
                            goodIO = 1
    except:
        pass
    if goodIO == 1:

        if data[0][5] == 'North':
            Location.append(1)
        elif data[0][5] == 'Central':
            Location.append(2)
        elif data[0][5] == 'South':
            Location.append(3)

        Age.append(data[0][0])
        Area.append(data[0][1])
        Carbon_all.append(data[0][2])
        Ele_annual_demand.append(data[0][3])
        Gas_annual_demand.append(data[0][4])

Energy_annual_demand = np.add(Ele_annual_demand, Gas_annual_demand)
Energy_intensity = np.divide(Energy_annual_demand, Area)
Carbon_intensity = np.divide(Carbon_all, Area)
Ele_intensity = np.divide(Ele_annual_demand, Area)
Gas_intensity = np.divide(Gas_annual_demand, Area)


d = {'Age':np.transpose(Age),
     'Area':np.transpose(Area),
     'Location':np.transpose(Location),
     'Carbon intensity':np.transpose(Carbon_intensity),
     'Ele annual demand': np.transpose(Ele_annual_demand),
     'Gas annual demand': np.transpose(Gas_annual_demand),
     'Gas intensity $kWh/ft^{2}$': np.transpose(Gas_intensity),
     'Electricity intensity $kWh/ft^{2}$':np.transpose(Ele_intensity)}
df = pd.DataFrame(d)
df.to_csv('out.csv')

#Plot Gas intensity vs location
plt.figure(1)
ax1 = sns.boxplot('Location', 'Gas intensity $kWh/ft^{2}$', data=df)
ax1.set(xticklabels=['North', 'Central', 'South'], title='Gas intensity of supermarkets in 2016-17')

#plot elec intensity vs location
plt.figure(2)
ax2 = sns.boxplot('Location', 'Electricity intensity $kWh/ft^{2}$', data=df)
ax2.set(xticklabels=['North', 'Central', 'South'], title='Gas intensity of supermarkets in 2016-17')

#Plot all variables against each other to find trend
sns.set(style="ticks", color_codes=True)
sns.pairplot(df, vars=["Age", "Area", "Carbon intensity", "Ele annual demand", "Gas annual demand"])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.1)

#plot Ele annual demand vs area

sns.set(style="darkgrid", color_codes=True)
sns.jointplot('Area', 'Ele annual demand', data=df, kind="reg", color="r", size=7)
plt.tight_layout()
#plot Gas annual demand vs area

sns.jointplot('Area', 'Gas annual demand', data=df, kind="reg", color="b", size=7)
plt.tight_layout()

plt.show()

