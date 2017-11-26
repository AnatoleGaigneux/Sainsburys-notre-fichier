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
import plotly.figure_factory as ff
import plotly as py
from collections import Counter

database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store_min = 2
id_store_max = 3000

#initialize matrices
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
CHP_size = []
financial_savings = []
closest_CHP_size = 10000
CHP_size_list = [35,50,90,100,110,125,135,150,165,185,210,230,310,375,400,425,500,530,770,850,460]
outliers = [['Store ID', 'Heat to Power ratio', 'Electricity Demand', 'MAC indicator', 'Payback']]

# =============================================================================
# CREATION OF SORTED DATA MATRIX 
# =============================================================================
# Go through all supermarkets of the business
# Get the Gas and ELec demand from database (and only where Elec demand is available --> reduces the number of stores considered)

for id_store in (j for j in range(id_store_min, id_store_max) if j != 2164):
    cur.execute(
        '''SELECT GD2016, ED2016, Carbon FROM Stores Where id= {vn1}'''.format(
            vn1=id_store))
    Index = cur.fetchall()
    Gas = np.array([elt[0] for elt in Index])
    Ele = np.array([elt[1] for elt in Index])
    carbon = np.array([elt[2] for elt in Index])

    if Ele != None:
        if Gas != None:
            MAC_calc = 5436.93*np.exp(-7.43635e-07*Ele[0])-541.107*np.exp(1.00852*Gas[0]/Ele[0])
            capex_calc = 0.0645517*Ele[0]+3507.16*Gas[0]/Ele[0]+444207
            payback_calc = 34.3592*np.exp(-8.42461e-07*Ele[0])+6.88398*np.exp(-1.02012*Gas[0]/Ele[0])
            carbon_savings_calc = 0.000260365*Ele[0]+542.343*Gas[0]/Ele[0]-429.405
            financial_savings_calc = 0.0343939*Ele[0]+20056.7*Gas[0]/Ele[0]-20709.5
            h2p_calc = Gas[0]/Ele[0]
            CHP_size_calc = 2689.29*np.exp(5.03651e-08*Ele[0])-2688.59*np.exp(-0.00382542*Gas[0]/Ele[0])
            for i in CHP_size_list:
                diff = abs(CHP_size_calc-i)
                if diff<abs(CHP_size_calc-closest_CHP_size):
                    closest_CHP_size = i
                    
                
            if h2p_calc >1.331: #create matrix of outliers and doesn't include them in timeline
                outliers.append([id_store, round(h2p_calc,2), int(Ele[0]), int(MAC_calc), round(payback_calc,1)])
            else:
                h2p.append(h2p_calc) 
                Ele_demand.append(Ele[0])
                payback.append(payback_calc) #from regression model
                carbon_savings.append(carbon_savings_calc) #from regression model
                BAU_carbon.append(carbon)
                capex.append(capex_calc) #from regression model
                stores.append(id_store)
                MAC.append(MAC_calc) #from regression model
                CHP_size.append(closest_CHP_size)
                financial_savings.append(financial_savings_calc)
                
                
        else: # When no Gas demand is available, average h2p ratio is used
            h2p.append(0.64502270251431926)
            Ele_demand.append(Ele[0])
            payback.append(34.3592*np.exp(-8.42461e-07*Ele[0])+6.88398*np.exp(-1.02012*0.64502270251431926))
            carbon_savings.append(0.000260365*Ele[0]+542.343*0.64502270251431926-429.405)
            BAU_carbon.append(carbon)
            capex.append(0.0645517*Ele[0]+3507.16*0.64502270251431926+444207)
            stores.append(id_store)
            MAC.append(5436.93*np.exp(-7.43635e-07*Ele[0])-541.107*np.exp(1.00852*0.64502270251431926))
            CHP_size_calc = 2689.29*np.exp(5.03651e-08*Ele[0])-2688.59*np.exp(-0.00382542*0.64502270251431926)
            for i in CHP_size_list:
                diff = abs(CHP_size_calc-i)
                if diff<abs(CHP_size_calc-closest_CHP_size):
                    closest_CHP_size = i
            CHP_size.append(closest_CHP_size)
            financial_savings.append(0.0343939*Ele[0]+20056.7*0.64502270251431926-20709.5)

# Data stored in matrix and sorted with respect to payback
Data =np.row_stack((payback, carbon_savings, capex, stores, MAC, CHP_size, financial_savings))
idx = np.argsort(Data[0])
Sorted_data = Data[:,idx]

#Create table for outliers
table = ff.create_table(outliers)
py.offline.plot(table, filename='Outliers table.html')


# =============================================================================
# CREATION OF INVESTMENT TIMELINE
# =============================================================================

#initialize matrices
cash_out = []
Carbon_saved_per_CHP = []
number_CHP_installed = []
num_viable_store = []    
Total_year_invest = []
Total_year_carbon = []
Total_year_cum_disc_cashflow =[]
Total_year_financial_savings = []

time = np.arange(2017,2051)
annual_invest = 9.9 #million p.a. #append for different scenarios

# Get potential carbon starting points for timeline
init_emissions = 1070000 #tCO2 during the year 2016-17 from report
BAU_emissions = np.sum(BAU_carbon) #tCO2 from supermarkets where CHP will be implemented(from master excel data)
        
#Initialize values
Cash_balance = 0
j = 0
n = 0
capex_needed = Sorted_data[2][j]/10**6

# Find number of CHP invested in for each year and resulting carbon savings and capex, assuming a certain annual investment
for i in range(0, len(time)) :
    Cash_balance = Cash_balance + annual_invest
    Individual_carbon_savings = []
    Individual_invest = []
    Individual_cum_disc_cashflow = []
    Individual_financial_saving = []
    try:
        while capex_needed<Cash_balance:
            cash_out = capex_needed
            Individual_invest.append(cash_out)
            Individual_carbon_savings.append(Sorted_data[1][j])
            Individual_cum_disc_cashflow.append(Sorted_data[4][j])
            Individual_financial_saving.append(Sorted_data[6][j])
            if Sorted_data[4][j]<0:
                n = n+1
            j= j+1
            capex_needed = Sorted_data[2][j]/10**6
            Cash_balance = Cash_balance -cash_out

    except:
        pass
#   print(Individual_cum_disc_cashflow)
    number_CHP_installed.append(j)
    num_viable_store.append(n)
    Total_year_invest.append(np.sum(Individual_invest))
    Total_year_carbon.append(np.sum(Individual_carbon_savings))
    Total_year_cum_disc_cashflow.append(np.average(Individual_cum_disc_cashflow))
    Total_year_financial_savings.append(np.sum(Individual_financial_saving))


Cum_invest = np.cumsum(Total_year_invest)
Cum_carbon_savings = np.cumsum(Total_year_carbon)
Cum_financial_savings = np.cumsum(Total_year_financial_savings)




#plot operating cost timeline
plt.figure(1)
BAU_operating_cost =  175776739 
plt.plot(time, BAU_operating_cost-Cum_financial_savings)
plt.ylabel('Stores operating cost timeline £')
plt.xlabel('time (years)')
            #plt.ylim(0,180000000)

Op_cost_reduction = Cum_financial_savings[-1]/BAU_operating_cost*100
print('Percentage operating cost reduced from 2016 total operating cost: %s %%' %Op_cost_reduction)

#Plot emissions timeline
plt.figure(2)
x = time
y = BAU_emissions-Cum_carbon_savings
plt.plot(x, y)
plt.xlabel('time (years)')
plt.ylabel('Emissions (tCO2)')
plt.ylim(0,550000)

#for i, txt in enumerate(np.transpose([np.diff(num_viable_store), np.diff(number_CHP_installed)])):
#    txt = ('{}/{}'.format(txt[0], txt[1]))
#    plt.annotate(txt, (x[i+1],y[i+1]))
#    

Emission_reduction_tot_footprint = Cum_carbon_savings[-1]/init_emissions*100
Emission_reduction_BAU = (Cum_carbon_savings[-1]/BAU_emissions*100)
print('Percentage emission reduced from 2016 total carbon footprint: %s %%' %Emission_reduction_tot_footprint)
print('Percentage emission reduced from BAU: %s %%' %Emission_reduction_BAU)

#bar chart presenting number of CHP installed per year
plt.figure(3)
ind = time[:-1] # number of bars
width = 0.6
p1 = plt.bar(ind, np.diff(num_viable_store), width, color='#d62728')
p2 = plt.bar(ind, np.diff(number_CHP_installed)-np.diff(num_viable_store), width, bottom=np.diff(num_viable_store))

plt.ylabel('Number of CHP installed')
plt.title('')
plt.xticks(ind)
plt.yticks(np.arange(0,5+max(np.diff(number_CHP_installed))))
plt.legend((p1, p2), ('Viable', 'Non viable'))
plt.show()

#Histogram showing distribution of type of CHP for first y years
plt.figure(4)
y_start=0
y_end=3
CHP_size_installed_by_year_y= Sorted_data[5][y_start:number_CHP_installed[y_end-1]]

a = Counter(np.array(CHP_size_installed_by_year_y,dtype=int))
a = np.transpose(np.array((list(a.items()))))
idx = np.argsort(a[0])
Sorted_a = a[:,idx]

ind = np.arange(len(Sorted_a[0])) # number of bars
width = 0.35

plt.bar(ind, Sorted_a[1], width, color='#d62728')

plt.ylabel('number of CHPs')
plt.xlabel('CHP size (MW)')
plt.title('Size distribution of CHP installed by year %d' %(2017+y_end))
plt.xticks(ind, (Sorted_a[0]))

plt.show()
