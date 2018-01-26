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
        
    def putTechPV(self, tech_id): 
        self.tech = pvtc.tech(tech_id)

        
    def OptiPVpanels(self, ):
        self.putTech(tech_id)
        
        tech_name = self.tech.tech_name
        tech_price = self.tech.tech_price*(1-ECA_value)          
        tech_lifetime = self.tech.lifetime
        
        for tech =1:10
              self.SimulatePVAllroof(tech)
           
            
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