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


for id_store in range(id_store_min, id_store_max ):
        cur.execute(
            '''SELECT GD2016, ED2016, Carbon FROM Stores Where id= {vn1}'''.format(
                vn1=id_store))
        Index = cur.fetchall()
        Gas = np.array([elt[0] for elt in Index])
        Ele = np.array([elt[1] for elt in Index])
        carbon = np.array([elt[2] for elt in Index])

        if (Gas or Ele) and (Gas and Ele) != None:

            h2p.append(Gas[0]/Ele[0])
            Ele_demand.append(Ele[0])
            payback.append(34.3592*np.exp(-8.42461e-07*Ele[0])+6.88398*np.exp(-1.02012*Gas[0]/Ele[0]))
            carbon_savings.append(0.000260365*Ele[0]+542.343*Gas[0]/Ele[0]-429.405)
            BAU_carbon.append(carbon)
            capex.append(0.0645517*Ele[0]+3507.16*Gas[0]/Ele[0]+444207)
            stores.append(id_store)
            Data =np.row_stack((payback, carbon_savings, capex, stores))    
            idx = np.argsort(Data[0])
            Sorted_data = Data[:,idx]

Emission_reduction_max = 0
        
for k in np.arange(1,10,0.01):
    time = np.arange(2017,2051)
    annual_invest = k #million p.a.
    
    
    init_emissions = 1070000 #tCO2 during the year 2016-17 from report
    BAU_emissions = np.sum(BAU_carbon) #tCO2 from supermarkets where CHP will be implemented(from master excel data)
        
    cash_out = []
    Carbon_saved_per_CHP = []
    number_CHP_installed = []
        
    Total_year_invest = []
    Total_year_carbon = []
        
    Cash_balance = 0
    j = 0
    capex_needed = Sorted_data[2][j]/10**6
    for i in range(0, len(time)) :
        Cash_balance = Cash_balance + annual_invest
        Individual_carbon_savings = []
        Individual_invest = []
        try:
            while capex_needed<Cash_balance:
                
                cash_out = capex_needed
                Carbon_saved_per_CHP = Sorted_data[1][j]
                Individual_invest.append(cash_out)
                Individual_carbon_savings.append(Carbon_saved_per_CHP)
                j= j+1
                capex_needed = Sorted_data[2][j]/10**6
                Cash_balance = Cash_balance -cash_out

                
        except:
            pass
        number_CHP_installed.append(j)
        Total_year_invest.append(np.sum(Individual_invest))
        Total_year_carbon.append(np.sum(Individual_carbon_savings))
    
        
    Cum_invest = np.cumsum(Total_year_invest)
    Cum_carbon_savings = np.cumsum(Total_year_carbon)
    
    Emission_reduction_tot_footprint = Cum_carbon_savings[-1]/init_emissions*100
    Emission_reduction_BAU = (Cum_carbon_savings[-1]/BAU_emissions*100)
    
    if Emission_reduction_BAU> Emission_reduction_max:
        Emission_reduction_max = Emission_reduction_BAU
        investment_max = k


x = time
y = BAU_emissions-Cum_carbon_savings
plt.plot(x, y, 'ro')
plt.xlabel('time (years)')
plt.ylabel('Emissions (tCO2)')
plt.ylim(0,450000)
for i, txt in enumerate(np.diff(number_CHP_installed)):
    plt.annotate(txt, (x[i],y[i]))
    
    
print('Percentage emission reduced from 2016 total carbon footprint: %s %%' %Emission_reduction_tot_footprint)
print('Percentage emission reduced from BAU: %s %%' %Emission_reduction_BAU)
print(investment_max)
