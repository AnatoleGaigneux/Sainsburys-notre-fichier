# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:48:09 2017

@author: Anatole
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly as py
from collections import Counter
from Non_Linear_regression_function import func1, func2, func3, func4, func5, popt1, popt2, popt3, popt4, popt5
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
Cum_disc_cashflow = []
CHP_size = []
financial_savings = []
Stores = []
closest_CHP_size = 10000
CHP_size_list = [35,50,90,100,110,125,135,150,165,185,210,230,310,375,400,425,500,530,770,850,460]
outliers = [['Store ID', 'Area (ft2)', 'Heat to Power ratio', 'Electricity Demand (GWh)', 'CDCF (thousand £/tCO2)', 'Payback (years)']]

ECA_value = 0.26
Hidden_costs = 353000
cur.execute('''SELECT price FROM Technologies''')
Tech_price= np.transpose(np.array(cur.fetchall()))[0]
m=0
g=0
# =============================================================================
# CREATION OF SORTED DATA MATRIX 
# =============================================================================
# Go through all supermarkets of the business
# Get the Gas and ELec demand from database (and only where Elec demand is available --> reduces the number of stores considered)

<<<<<<< HEAD
=======
    


>>>>>>> ab7c22c72b825faafb527b172d703bb2175d58aa
for id_store in (j for j in range(id_store_min, id_store_max) if j != (2164 and 490 and 2019 and 2020 and 2035 and 2043 and 2107 and 2116 and 2500)): # 2164 is an outlier not used in the regression model, all the rest are convienience stores
    cur.execute(
        '''SELECT GD2016, ED2016, Carbon, Area FROM Stores Where id= {vn1}'''.format(
            vn1=id_store))
    Index = cur.fetchall()
    
    Gas = np.array([elt[0] for elt in Index])  #kWh
    Ele = np.array([elt[1] for elt in Index]) #kWh
    carbon = np.array([elt[2] for elt in Index])
    Area = np.array([elt[3] for elt in Index])  
    if Ele != None:
        if Gas != None:
            Ele=Ele/10**6 #GWh
            Gas=Gas/10**6 #GWh
            Cum_disc_cashflow_calc = func4([Ele[0], Gas[0]/Ele[0]],*popt4)  #thousand £ (NPV-capex)
            payback_calc = func1([Ele[0], Gas[0]/Ele[0]],*popt1) #years
            carbon_savings_calc = func2([Ele[0], Gas[0]/Ele[0]],*popt2)#tCO2
            financial_savings_calc = func3([Ele[0], Gas[0]/Ele[0]],*popt3)  #Thousands of £
            h2p_calc = Gas[0]/Ele[0]
            CHP_size_calc = func5(Ele[0],*popt5) # in MW
            k=0
            for i in CHP_size_list:
                diff = abs(CHP_size_calc-i)
                if diff<abs(CHP_size_calc-closest_CHP_size):
                    closest_CHP_size = i
                    capex_calc = (Tech_price[k]*(1-ECA_value)+Hidden_costs)/10**6 #million £
                k=k+1
            if h2p_calc >1.331: #create matrix of outliers and doesn't include them in timeline
                outliers.append([id_store, Area[0], round(h2p_calc,2), round(Ele[0],2), int(Cum_disc_cashflow_calc), round(payback_calc,1)])
            else:
                h2p.append(h2p_calc) 
                Ele_demand.append(Ele[0])
                payback.append(payback_calc) #from regression model
                carbon_savings.append(carbon_savings_calc) #from regression model
                BAU_carbon.append(carbon[0])
                capex.append(capex_calc) #from regression model
                stores.append(id_store)
                Cum_disc_cashflow.append(Cum_disc_cashflow_calc) #from regression model
                CHP_size.append(closest_CHP_size)
                financial_savings.append(financial_savings_calc)
                Stores.append(id_store)
                m=m+1
        else: # When no Gas demand is available, average h2p ratio is used
            Ele=Ele/10**6 #GWh
            h2p.append(0.6450)
            Ele_demand.append(Ele[0])
            payback.append(func1([Ele[0],0.6450],*popt1))
            carbon_savings.append(func2([Ele[0],0.6450],*popt2))
            BAU_carbon.append(carbon[0])
            stores.append(id_store)
            Cum_disc_cashflow.append(func4([Ele[0],0.6450],*popt4))
            CHP_size_calc = func5(Ele[0],*popt5)
            k=0
            for i in CHP_size_list:
                diff = abs(CHP_size_calc-i)
                if diff<abs(CHP_size_calc-closest_CHP_size):
                    closest_CHP_size = i
                    capex_calc = (Tech_price[k]*(1-ECA_value)+Hidden_costs)/10**6 #million £
                k=k+1
            CHP_size.append(closest_CHP_size)
            capex.append(capex_calc)
            financial_savings.append(func3([Ele[0],0.6450],*popt3))
            Stores.append(id_store)
            g=g+1
# Data stored in matrix and sorted with respect to payback
Data =np.row_stack((payback, carbon_savings, capex, stores, Cum_disc_cashflow, CHP_size, financial_savings, Stores, BAU_carbon))
idx = np.argsort(Data[0])
Sorted_data = Data[:,idx]

#Create table for outliers
table = ff.create_table(outliers)
py.offline.plot(table, filename='Outliers table.html')


# =============================================================================
# CREATION OF INVESTMENT TIMELINE
# =============================================================================

#initialize matrices uses in plots
# Get potential carbon starting points for timeline

Y= []
y = []
Y1 = []
Y2= []
Y3 =[]

Sorted_a = []
Op_cost_reduction = []
Emission_reduction_tot_footprint = []
Emission_reduction_BAU = []
All_payback = []

annual_invest_list = [7.88,10.59]
for i in annual_invest_list :
    #initialize matrices
    cash_out = []
    Carbon_saved_per_CHP = []
    number_CHP_installed = [0]
    num_viable_store = [0]    
    Total_year_invest = []
    Total_year_carbon = []
    Total_year_cum_disc_cashflow =[]
    Total_year_financial_savings = []
    Average_year_payback = []
    
    time = np.arange(2017,2051)
    annual_invest = i #million p.a. #append for different scenarios
            
    #Initialize values
    Cash_balance = 0
    j = 0
    n = 0
    capex_needed = Sorted_data[2][j]
    
    # Find number of CHP invested in for each year and resulting carbon savings and capex, assuming a certain annual investment
    for i in range(0, len(time)) :
        Cash_balance = Cash_balance + annual_invest
        Individual_carbon_savings = []
        Individual_invest = []
        Individual_cum_disc_cashflow = []
        Individual_financial_saving = []
        Individual_payback =[]
        try:
            while capex_needed<Cash_balance:
                cash_out = capex_needed
                Individual_invest.append(cash_out)
                Individual_carbon_savings.append(Sorted_data[1][j])
                Individual_cum_disc_cashflow.append(Sorted_data[4][j])
                Individual_financial_saving.append(Sorted_data[6][j])
                Individual_payback.append(Sorted_data[0][j])
                if Sorted_data[4][j]>0:
                    n = n+1
                j= j+1
                capex_needed = Sorted_data[2][j]
                Cash_balance = Cash_balance -cash_out
    
        except:
            pass
        number_CHP_installed.append(j)
        num_viable_store.append(n)
        Total_year_invest.append(np.sum(Individual_invest))
        Total_year_carbon.append(np.sum(Individual_carbon_savings))
        Total_year_cum_disc_cashflow.append(np.average(Individual_cum_disc_cashflow))
        Total_year_financial_savings.append(np.sum(Individual_financial_saving))
        if np.average(Individual_payback)<10:
            Average_year_payback.append(round(np.average(Individual_payback),1))
        else:
            Average_year_payback.append(int(np.average(Individual_payback)))
    
    Cum_invest = np.cumsum(Total_year_invest)
    Cum_carbon_savings = np.cumsum(Total_year_carbon)
    Cum_financial_savings = np.cumsum(Total_year_financial_savings)
    All_payback.append(Average_year_payback)
    
    BAU_operating_cost =  174656.940 #from master excel sheet 2015 (thousands of £)
    init_emissions = 1043000 #tCO2 during the year 2015 from buildings (master excel data)
    BAU_emissions = np.sum(Sorted_data[8][0:j]) #tCO2 from supermarkets where CHP will be implemented(from master excel data)

    #Create matrices to plot 
    Y.append((BAU_operating_cost-Cum_financial_savings)/1000)  # million £
    y.append((init_emissions-Cum_carbon_savings)/1000) # ktCO2
    Op_cost_reduction.append(round(Cum_financial_savings[-1]/BAU_operating_cost*100,1))
    Emission_reduction_tot_footprint.append(round(Cum_carbon_savings[-1]/init_emissions*100,1))
    Emission_reduction_BAU.append((Cum_carbon_savings[-1]/BAU_emissions*100))
    Y1.append(np.diff(num_viable_store))
    Y2.append(np.diff(number_CHP_installed)-np.diff(num_viable_store))
    Y3.append(np.diff(number_CHP_installed))

    y_start=0
    y_end=1
    CHP_size_installed_by_year_y= Sorted_data[5][y_start:number_CHP_installed[y_end]]
    Stores_with_CHP_by_year_y = Sorted_data[7][y_start:number_CHP_installed[y_end]]
    if annual_invest == annual_invest_list[0]:
        for i in range(0,len(Stores_with_CHP_by_year_y)):            
            print(Stores_with_CHP_by_year_y[i],CHP_size_installed_by_year_y[i],Sorted_data[0][i],Sorted_data[6][i]/(Sorted_data[2][i]*1000))
    a = Counter(np.array(CHP_size_installed_by_year_y,dtype=int))
    a = np.transpose(np.array((list(a.items()))))
    idx = np.argsort(a[0])
    Sorted_a.append(a[:,idx])

# =============================================================================
# PLOTS
# =============================================================================
X=time

#plot operating cost timeline
plt.figure(6)

plt.plot(X, Y[0],'dodgerblue', label='annual investment: £%s million' %annual_invest_list[0])
plt.plot(X, Y[1],'blue', label='annual investment: £%s million' %annual_invest_list[1])
plt.ylabel('Stores operating cost timeline (million £)')
plt.xlabel('time (years)')
plt.legend(loc=3)


plt.annotate('Operating Cost\n remaining \n%s %%' %(100-Op_cost_reduction[0]), (X[-1],Y[0][-1]+3.5))
plt.annotate('Operating Cost\n remaining \n%s %%' %(100-Op_cost_reduction[1]), (X[-1],Y[1][-1]+1))


for i, txt in enumerate(All_payback[0]):
    plt.annotate(txt, (X[i],Y[0][i]*1.01), bbox={'facecolor':'white', 'pad':3})
for i, txt in enumerate(All_payback[1]):
    plt.annotate(txt, (X[i]-1,Y[1][i]*0.985), bbox={'facecolor':'white', 'pad':3})
    


#Plot emissions timeline
plt.figure(7)
plt.plot(X, y[0],'dodgerblue', label='annual investment: £%s million' %annual_invest_list[0])
plt.plot(X, y[1],'blue', label='annual investment: £%s million' %annual_invest_list[1])
plt.xlabel('time (years)')
plt.ylabel('Emissions (ktCO2)')
plt.legend(loc=3)

plt.annotate('Emissions\n remaining:\n%s %%' %(100-Emission_reduction_tot_footprint[0]), (X[-1],y[0][-1]))
plt.annotate('Emissions\n remaining:\n%s %%' %(100-Emission_reduction_tot_footprint[1]), (X[-1],y[1][-1]))

#bar chart presenting number of CHP installed per year
plt.figure(8)
ind = time+1 # number of bars
width = 0.6
p1 = plt.bar(ind, Y1[0], width, color='#d62728', hatch = '//')
p2 = plt.bar(ind, Y2[0], width, bottom=Y1[1], color= 'navy', hatch ='///')

plt.ylabel('Number of CHP installed')
plt.title('')
plt.xticks(ind)
plt.yticks(np.arange(0,max(np.diff(number_CHP_installed))))
plt.legend((p1, p2), ('Financially benificial', 'Financially non beneficial'))
plt.show()

#Histogram showing distribution of type of CHP for first y years
plt.figure(9)
ind = range(0,len(Sorted_a[1][0])) # number of bars
width = 0.3

plt.bar(ind, Sorted_a[1][1], width, color='navy',label='annual investment: £%s million' %annual_invest_list[1])      

plt.ylabel('number of CHPs')
plt.xlabel('CHP size (MW)')
plt.title('Size distribution of CHP installed by year %d' %(2017+y_end))
plt.xticks(ind, (Sorted_a[1][0]))
plt.legend(loc=1)

plt.show()

# =============================================================================
# Max error of using average H2P ratio
# =============================================================================
#Payback
max_payback_error=0
for Elec in np.arange(0.77,4.58,0.01):
    for Heat2Power in np.arange(0.13,1.33,0.01):
        payback_error = abs(24.9*np.exp(-0.907*Elec)+4.96*np.exp(-1.09*Heat2Power)-(24.9*np.exp(-0.907*Elec)+4.96*np.exp(-1.09*0.6450))) #years
        if payback_error>max_payback_error:
            max_payback_error =payback_error
            max_payback_relative_error = payback_error*100/(24.9*np.exp(-0.907*Elec)+4.96*np.exp(-1.09*0.6450))

#Cum Discounted Cash Flow error
max_CDCF_error=0
for Elec in np.arange(0.77,4.58,0.01):
    for Heat2Power in np.arange(0.13,1.33,0.01):
        CDCF_error = abs(419*Elec+291*Heat2Power-762-(419*Elec+291*0.6450-762)) #thousand £ (NPV-capex)
        if CDCF_error>max_CDCF_error:
            max_CDCF_error =CDCF_error
            max_CDCF_relative_error = CDCF_error*100/(419*Elec+291*0.6450-762)

#carbon savings error
max_carbon_savings_error=0
for Elec in np.arange(0.77,4.58,0.01):
    for Heat2Power in np.arange(0.13,1.33,0.01):
        carbon_savings_error = abs(367*Elec+512*Heat2Power-439-(367*Elec+512*0.6450-439))#tCO2
        if carbon_savings_error>max_carbon_savings_error:
            max_carbon_savings_error = carbon_savings_error
            max_carbon_savings_relative_error =carbon_savings_error*100/(367*Elec+512*0.6450-439)
            
#operating cost savings error
max_financial_savings_error=0
for Elec in np.arange(0.77,4.58,0.01):
    for Heat2Power in np.arange(0.13,1.33,0.01):
        financial_savings_error = abs(53.0*Elec+33.0*Heat2Power-35.0-(53.0*Elec+33.0*0.6450-35.0))  #Thousands of £
        if financial_savings_error>max_financial_savings_error:
            max_financial_savings_error = financial_savings_error
            max_financial_savings_realtive_error = financial_savings_error*100/(53.0*Elec+33.0*0.6450-35.0)
            
#print(max_payback_error, max_payback_relative_error)
#print(max_CDCF_error, max_CDCF_relative_error)
#print(max_carbon_savings_error,max_carbon_savings_relative_error)
#print(max_financial_savings_error, max_financial_savings_realtive_error)

