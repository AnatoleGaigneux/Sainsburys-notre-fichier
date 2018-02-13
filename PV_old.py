# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:24:43 2018

@author: Alex  
"""
import sqlite3
import numpy as np
import datetime
import calendar
import os
import sys
scriptpath = "Common" # This is written in the Windows way of specifying paths, hopefully it works on Linux?
sys.path.append(os.path.abspath(scriptpath))
import Common.classStore as st # Module is in seperate folder, hence the elaboration
import Common.classPVTech as pvtc
#from pyomo.environ import * # Linear programming module
#import pyomo as pyo
#import pyomo.environ as pyo
#from pyomo.environ import *
#from pyomo.opt import SolverFactory # Solver
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
<<<<<<< HEAD
        self.discount_rate = 0.09
        self.roof_max_weight =
        self.roof_area=

=======
        #put self. back 
        discount_rate = 0.09
        roof_max_weight = 16 #(kg/m2)
        roof_area= 400 #m2
        hidden_cost=2000 #£
        Roof_space_coeff=0.6 
        roof_available_area= roof_area*Roof_space_coeff #m2

        
<<<<<<< HEAD
        
=======

>>>>>>> 8901107b834b5d0c09f99a6692f2de2c9649529a
>>>>>>> a0df21a56162ff15b5f73360a125994640aaac31
    def putTechPV(self, tech_id): 
        self.tech = pvtc.tech(tech_id)

        
<<<<<<< HEAD
    def OptiPVpanels(self, method = None, tech_range = None, time_start = None, time_stop = None, table_string = None, ECA_value = 0, uncertainty = None, mod = None):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string)            
        if tech_range is not None:
            array_tech = tech_range
        else:
            array_tech = range(1,4) 
            
            
        opti_savings=0
        
    
=======
    def OptiPVpanels(self, tech_range):
<<<<<<< HEAD
        tech_opex = 0
        Elec_grid = 0
        Elec_surplus = 0
        for tech_id in tech_range:
            #initialize

=======

    tech_opex = 0
    Elec_grid = 0
    Elec_surplus = 0
    for tech_id in tech_range:
            #initialize


>>>>>>> 8901107b834b5d0c09f99a6692f2de2c9649529a

        for tech_id in tech_range:
            # initialize
            discount_rate = 0.09
            roof_max_weight = 8 #(kg/m2)
            roof_area= 400 #m2

            
            tech_opex = 0
            Elec_grid = 0
            Elec_surplus = 0
<<<<<<< HEAD
            self.putTech(tech_id)
            tech_name = self.tech.tech_name
            tech_price = self.tech.tech_price*(1-ECA_value)          
            tech_lifetime = self.tech.lifetime
            tech_eff= self.tech.eff
            tech_area=self.tech.area
            tech_weight=self.tech.weight
            Store_demand = self.store.d_ele

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
            Opex_savings = Total_Elec_prod*tech_lifetime
            Opex=gas_demand*gas_price+(Elec_grid-Elec_surplus)*price_elec - policies*Total_Elec_prod
            if Opex_savings>tech_opex:
                best_tech=tech_name
            else:
                tech_opex=opex_savings
            return(best_tech,Opex_savings,capex,Opex)


            Opex_savings = 
            Opex=gas_demand*gas_price+(Elec_grid-elec_surplus)*price_elec - policies*Total_Elec_prod

            
    def calculate_financials(self, discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex):
=======
            #self.putTech(tech_id)
            tech_name = 'Mono-Si'
            tech_price = 0.048 #(£/Wp,yr)
            tech_lifetime = 20
            tech_eff = 0.1807
            tech_area = 1.65 #(m2)
            tech_weight = 19.1 #(kg)
            #Store_demand = self.store.d_ele

>>>>>>> a0df21a56162ff15b5f73360a125994640aaac31

        for id_tech_index in array_tech:
            tech_id = id_tech_index
            self.putTech(tech_id)
            tech_name = self.tech.PVtech_name
            tech_price = self.tech.PV_capex #(£/Wp,yr)
            tech_lifetime = self.tech.PV_lifetime
            tech_eff = self.tech.PV_eff
            tech_area = self.tech.PV_Area #(m2)
            tech_weight = self.tech.PV_Weight #(kg)
            #Store_demand = self.store.d_ele
    
    
            if tech_weight / tech_area < roof_max_weight:
                irradiance = np.array([0,0,1,2,3,6,3,2,1,0,0])
                Indiv_Elec_prod = tech_eff * np.array(irradiance)
                N_panel = int(roof_available_area / tech_area)
                Total_Elec_prod = N_panel * Indiv_Elec_prod
                panel_price = 0.437*293
                Store_demand = np.array([0, 0, 3, 5, 6, 7, 23, 34, 120, 23, 0])
                
                mask0=(Total_Elec_prod<Store_demand).astype(int)
                mask1=(Total_Elec_prod>Store_demand).astype(int)
                Elec_grid = mask0*(Store_demand - Total_Elec_prod)
                Elec_surplus=mask1*(Total_Elec_prod-Store_demand)
    
                # Costs
                Total_capex = N_panel * panel_price + hidden_cost 
                Opex_savings = Total_Elec_prod *(elec_price +policies) 
                op_cost_HH_pound = (Elec_grid-Elec_surplus)*elec_price + gas_demand*gas_price
                if np.sum(Opex_savings) > opti_savings: 
                    best_tech = tech_name
                    opti_savings= np.sum(Opex_savings)
                else:
                    pass 
            return (best_tech, Opex_savings, Total_capex)

        def calculate_financials(self, discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex):
<<<<<<< HEAD
            numb_years = 10 #(self.time_stop-self.time_start)/2/24/365
            year_op_cost = sum(op_cost_HH_pound)/numb_years
            year_BAU_cost = 1330 #sum(BAU_op_cost_HH_pound)/numb_years
=======
>>>>>>> 8901107b834b5d0c09f99a6692f2de2c9649529a
>>>>>>> a0df21a56162ff15b5f73360a125994640aaac31
            year_savings = year_BAU_cost - year_op_cost
            payback = Total_capex / year_savings
            ann_capex = -np.pmt(discount_rate, tech_lifetime, Total_capex)
            year_cost = year_op_cost + ann_capex
            NPV5_op_cost = -np.npv(discount_rate, np.array([year_cost] * 5))
            NPV5_BAU_cost = -np.npv(discount_rate, np.array([year_BAU_cost] * 5))
            NPV5savings = NPV5_op_cost - NPV5_BAU_cost
            ROI = year_savings / Total_capex
            Const = (1 - (1 + discount_rate) ** (-tech_lifetime)) / discount_rate
            Cum_disc_cash_flow = -Total_capex + Const * year_savings
            return (year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow)







            # store best value


            # output optimum and KPIs

        def SimulatePVonAllRoof(self, tech):
            ####

            # get tech data
            # connect to database
            # retrieve data
            # store data

            self.PVtech.eff = 9


            ###



            ###
            # find number of panels
            # find output per panels by multpy eff times irradiance
            # find totla electricity
            # find cost reduction
            # find revenue from policy

            ####


            ###
            # output some indicators, savings, capex, payback time, irr
            ####