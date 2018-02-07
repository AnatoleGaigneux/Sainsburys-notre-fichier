# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:24:43 2018

@author: Anatole
"""
import sqlite3
import numpy as np
import datetime
import calendar
import os
import sys
scriptpath = "..\\Common\\" # This is written in the Windows way of specifying paths, hopefully it works on Linux?
sys.path.append(os.path.abspath(scriptpath))
import Common.classStore as st # Module is in seperate folder, hence the elaboration
import Common.classTech as tc
#from pyomo.environ import * # Linear programming module
#import pyomo as pyo
#import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory # Solver
import time # To time code, simply use start = time.clock() then print start - time.clock()


class PVproblem:
   
    
    def __init__(self, store_id):           
        self.store = st.store(store_id)
        default_initial_time = datetime.datetime(2016,1,1)
        default_final_time = datetime.datetime(2017,1,1)
        self.time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
        self.store.getSimpleDemand(self.time_start, self.time_stop)
        self.store.getWeatherData(self.time_start, self.time_stop)
        self.discount_rate = 0.09
        self.roof_max_weight =
        self.roof_area= 
        
    def putTechPV(self, tech_id): 
        self.tech = pvtc.tech(tech_id)

        
    def OptiPVpanels(self, tech_range):

        for tech_id in tech_range:
            #initialize
            Elec_grid = 0
            Elec_surplus = 0
            self.putTech(tech_id)
            tech_name = self.tech.tech_name
            tech_price = self.tech.tech_price*(1-ECA_value)          
            tech_lifetime = self.tech.lifetime
            tech_eff= self.tech.eff
            tech_area=self.tech.area
            tech_weight=self.tech.weight
            
            if tech_weight/tech_area < roof_max_weight:
                Indiv_Elec_prod = tech_eff*irradiance
                N_panel = int(roof_area/tech_area)
                Total_Elec_prod = N_panel*Indiv_Elec_prod
                Store_demand = self.store.d_ele
                if Total_Elec_prod> Store_demand:
                    Elec_surplus = Total_Elec_prod- Store_demand
                else:
                    Elec_grid = Store_demand - Total_Elec_prod
                    
            #Costs
            capex=N_panel*tech_price
            Opex_savings = 
            Opex=gas_demand*gas_price+(Elec_grid-elec_surplus)*price_elec - policies*Total_Elec_prod

            
    def calculate_financials(self, discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex):
            year_savings = year_BAU_cost - year_op_cost
            payback = Total_capex/year_savings           
            ann_capex = -np.pmt(discount_rate, tech_lifetime, Total_capex)            
            year_cost = year_op_cost + ann_capex
            NPV5_op_cost  = -np.npv(discount_rate, np.array([year_cost]*5))
            NPV5_BAU_cost = -np.npv(discount_rate, np.array([year_BAU_cost]*5)) 
            NPV5savings = NPV5_op_cost - NPV5_BAU_cost
            ROI = year_savings/Total_capex
            Const = (1-(1+discount_rate)**(-tech_lifetime))/discount_rate            
            Cum_disc_cash_flow = -Total_capex + Const*year_savings
            return(year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow)
            
            
            

                
        
            
         #store best value
        
         
         #output optimum and KPIs
        
    def SimulatePVonAllRoof(self, tech):
        ####
       
        #get tech data
        #connect to database
        #retrieve data
        #store data
       
        self.PVtech.eff = 9
 
       
        ###
       
        
        
        ###
        #find number of panels
        #find output per panels by multpy eff times irradiance
        #find totla electricity
        #find cost reduction
        #find revenue from policy       
        
        ####
       
        
        ###
        #output some indicators, savings, capex, payback time, irr
        ####