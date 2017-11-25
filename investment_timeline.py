# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:48:09 2017

@author: Anatole
"""

import sqlite3
import Common.classStore as st
import Common.classTech as tc
import Solvers.classCHPProblemnew as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
import re

database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store_min = 2
id_store_max = 3000

payback = []
h2p = []
Area = []
Ele_demand = []
Gas_demand = []
Age = []
carbon_savings = []
capex = []
BAU_carbon = []
stores = []
MAC = []


for id_store in (j for j in range(id_store_min, id_store_max) if j != 2164):
    cur.execute(
        '''SELECT GD2016, ED2016, Carbon FROM Stores Where id= {vn1}'''.format(
            vn1=id_store))
    Index = cur.fetchall()
    Gas = np.array([elt[0] for elt in Index])
    Ele = np.array([elt[1] for elt in Index])
    carbon = np.array([elt[2] for elt in Index])

#        if (Gas or Ele) and (Gas and Ele) != None:
    if Ele != None:
        if Gas != None:
            h2p.append(Gas[0]/Ele[0])
            Ele_demand.append(Ele[0])
            payback.append(34.3592*np.exp(-8.42461e-07*Ele[0])+6.88398*np.exp(-1.02012*Gas[0]/Ele[0]))
            carbon_savings.append(0.000260365*Ele[0]+542.343*Gas[0]/Ele[0]-429.405)
            BAU_carbon.append(carbon)
            capex.append(0.0645517*Ele[0]+3507.16*Gas[0]/Ele[0]+444207)
            stores.append(id_store)
            MAC.append(5436.93*np.exp(-7.43635e-07*Ele[0])-541.107*np.exp(1.00852*Gas[0]/Ele[0]))

        else:
            h2p.append(0.64502270251431926)
            Ele_demand.append(Ele[0])
            payback.append(34.3592*np.exp(-8.42461e-07*Ele[0])+6.88398*np.exp(-1.02012*0.64502270251431926))
            carbon_savings.append(0.000260365*Ele[0]+542.343*0.64502270251431926-429.405)
            BAU_carbon.append(carbon)
            capex.append(0.0645517*Ele[0]+3507.16*0.64502270251431926+444207)
            stores.append(id_store)
            MAC.append(5436.93*np.exp(-7.43635e-07*Ele[0])-541.107*np.exp(1.00852*0.64502270251431926))
    
Data =np.row_stack((payback, carbon_savings, capex, stores,MAC))
idx = np.argsort(Data[0])
Sorted_data = Data[:,idx]

Emission_reduction_max = 0
        

time = np.arange(2017,2051)
annual_invest = 9.9 #million p.a.
    
init_emissions = 1070000 #tCO2 during the year 2016-17 from report
BAU_emissions = np.sum(BAU_carbon) #tCO2 from supermarkets where CHP will be implemented(from master excel data)
        
cash_out = []
Carbon_saved_per_CHP = []
number_CHP_installed = []
num_viable_store = []    
Total_year_invest = []
Total_year_carbon = []
Total_year_cum_disc_cashflow =[]
        
Cash_balance = 0
j = 0
n = 0
capex_needed = Sorted_data[2][j]/10**6
for i in range(0, len(time)) :
    Cash_balance = Cash_balance + annual_invest
    Individual_carbon_savings = []
    Individual_invest = []
    Individual_cum_disc_cashflow = []
    try:
        while capex_needed<Cash_balance:
                
            cash_out = capex_needed
            Individual_invest.append(cash_out)
            Individual_carbon_savings.append(Sorted_data[1][j])
            if -10000<Sorted_data[4][j]<10000:
                Individual_cum_disc_cashflow.append(Sorted_data[4][j])
            if Sorted_data[4][j]<0:
                n = n+1
            j= j+1
            capex_needed = Sorted_data[2][j]/10**6
            Cash_balance = Cash_balance -cash_out

    except:
        pass
        
    number_CHP_installed.append(j)
    num_viable_store.append(n)
    Total_year_invest.append(np.sum(Individual_invest))
    Total_year_carbon.append(np.sum(Individual_carbon_savings))
    Total_year_cum_disc_cashflow.append(np.average(Individual_cum_disc_cashflow))


Cum_invest = np.cumsum(Total_year_invest)
Cum_carbon_savings = np.cumsum(Total_year_carbon)

Emission_reduction_tot_footprint = Cum_carbon_savings[-1]/init_emissions*100
Emission_reduction_BAU = (Cum_carbon_savings[-1]/BAU_emissions*100)


plt.figure(1) #investment timeline (carbon)
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(time, Total_year_cum_disc_cashflow)
plt.ylabel('Yearly average cumulative discounted cashflow')

x = time
y = BAU_emissions-Cum_carbon_savings
axarr[1].plot(x, y, 'ro')
plt.xlabel('time (years)')
plt.ylabel('Emissions (tCO2)')
axarr[1].ylim(0,550000)
for i, txt in enumerate(np.transpose([np.diff(num_viable_store), np.diff(number_CHP_installed)])):
    txt = ('{}/{}'.format(txt[0], txt[1]))
    axarr[1].annotate(txt, (x[i+1],y[i+1]))
    


print('Percentage emission reduced from 2016 total carbon footprint: %s %%' %Emission_reduction_tot_footprint)
print('Percentage emission reduced from BAU: %s %%' %Emission_reduction_BAU)

plt.figure(1)
ind = time[:-1] # number of bars
width = 0.6
p1 = plt.bar(ind, np.diff(num_viable_store), width, color='#d62728')
p2 = plt.bar(ind, np.diff(number_CHP_installed)-np.diff(num_viable_store), width, bottom=np.diff(num_viable_store))

plt.ylabel('Number of CHP installed')
plt.title('')
plt.xticks(ind)
plt.yticks(np.arange(0,max(np.diff(number_CHP_installed))))
plt.legend((p1, p2), ('Viable', 'Non viable'))
