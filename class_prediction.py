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

class investment_model:
    
    def __init__(self,id_store):
        
        store_id_range = np.arange(1,3000)
        stores_not_included = [2164, 490, 2019, 2020, 2035, 2043, 2107, 2116, 2500]
        self.store_id_range = np.delete(store_id_range, [x-1 for x in stores_not_included])
        self.closest_CHP_size = 10000
        self.CHP_size_list = [35,50,90,100,110,125,135,150,165,185,210,230,310,375,400,425,500,530,770,850,460]
        self.outliers = [['Store ID', 'Area (ft2)', 'Heat to Power ratio', 'Electricity Demand (GWh)', 'CDCF (thousand £/tCO2)', 'Payback (years)']]
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        cur.execute('''SELECT price FROM Technologies''')
        self.ECA_value = 0.26
        self.Hidden_costs = 353000
        self.Tech_price= np.transpose(np.array(cur.fetchall()))[0]

# =============================================================================
#  PREDICTIONS FROM REGRESSION
# =============================================================================

# Get the Gas and ELec demand from database (and only where Elec demand is available --> icreases the number of stores considered)
    def regression_prediction(self, id_store):
        
        #initialize matrices
        self.payback = []
        self.h2p = []
        self.Area = []
        self.Ele_demand = []
        self.Gas_demand = []
        self.Age = []
        self.carbon_savings = []
        self.capex = []
        self.BAU_carbon = []
        self.stores = []
        self.Cum_disc_cashflow = []
        self.CHP_size = []
        self.financial_savings = []
        self.Stores = []
        self.num_with_real_h2p=0
        self.num_with_approx_h2p=0
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()

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
                for i in self.CHP_size_list:
                    diff = abs(CHP_size_calc-i)
                    if diff<abs(CHP_size_calc-self.closest_CHP_size):
                        self.closest_CHP_size = i
                        capex_calc = (self.Tech_price[k]*(1-self.ECA_value)+self.Hidden_costs)/10**6 #million £
                    k=k+1
                if h2p_calc >1.331: #create matrix of outliers and doesn't include them in timeline
                    self.outliers.append([id_store, Area[0], round(h2p_calc,2), round(Ele[0],2), int(Cum_disc_cashflow_calc), round(payback_calc,1)])
                else:
                    self.h2p.append(h2p_calc) 
                    self.Ele_demand.append(Ele[0])
                    self.payback.append(payback_calc) #from regression model
                    self.carbon_savings.append(carbon_savings_calc) #from regression model
                    self.BAU_carbon.append(carbon[0])
                    self.capex.append(capex_calc) #from regression model
                    self.stores.append(id_store)
                    self.Cum_disc_cashflow.append(Cum_disc_cashflow_calc) #from regression model
                    self.CHP_size.append(self.closest_CHP_size)
                    self.financial_savings.append(financial_savings_calc)
                    self.Stores.append(id_store)
                    self.num_with_real_h2p=self.num_with_real_h2p+1
            else: # When no Gas demand is available, average h2p ratio is used
                Ele=Ele/10**6 #GWh
                self.h2p.append(0.6450)
                self.Ele_demand.append(Ele[0])
                self.payback.append(func1([Ele[0],0.6450],*popt1))
                self.carbon_savings.append(func2([Ele[0],0.6450],*popt2))
                self.BAU_carbon.append(carbon[0])
                self.stores.append(id_store)
                self.Cum_disc_cashflow.append(func4([Ele[0],0.6450],*popt4))
                CHP_size_calc = func5(Ele[0],*popt5)
                k=0
                for i in self.CHP_size_list:
                    diff = abs(CHP_size_calc-i)
                    if diff<abs(CHP_size_calc-self.closest_CHP_size):
                        self.closest_CHP_size = i
                        capex_calc = (self.Tech_price[k]*(1-self.ECA_value)+self.Hidden_costs)/10**6 #million £
                    k=k+1
                self.CHP_size.append(self.closest_CHP_size)
                self.capex.append(self.capex_calc)
                self.financial_savings.append(func3([Ele[0],0.6450],*popt3))
                self.Stores.append(id_store)
                self.num_with_approx_h2p=self.num_with_approx_h2p+1
        return(self.payback)
