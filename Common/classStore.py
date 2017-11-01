# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:09:32 2017

@author: nl211
"""
import sqlite3
import numpy as np


#database_path = "C:\\Users\\GOBS\\Dropbox\\Uni\Other\\UROP - Salvador\\Niccolo_project\\Code\\Sainsburys.sqlite" # Path to database file
database_path = ".\\Sainsburys.sqlite" # Path to database file


class store:
    
    def __init__(self, store_id):
        self.store_id = store_id
        self.HH_open = 14
        self.HH_close = 47
        try:    
            conn = sqlite3.connect(database_path)
            cur = conn.cursor()
        except ValueError:
            print("Cannot connect to database")
        try:    
            cur.execute("SELECT DNO, name FROM Stores WHERE id = ?", (store_id,))
            dummy = cur.fetchall()
            self.DNO = dummy[0][0]    
            self.Voltage = 1   #all stores are assumed to have low voltage sub connection.
            self.name = dummy[0][1]   
        except ValueError: 
            print("Cannot retrieve store data")
        try:    
            cur.execute("SELECT HH_WD_open,  HH_WD_close,    HH_Sat_open,    HH_Sat_close,    HH_Sun_open,   HH_Sun_close FROM Stores WHERE id = ?", (store_id,))
            dummy = cur.fetchall()
            self.HH_WD_open = dummy[0][0]    
            self.HH_WD_close = dummy[0][1]
            self.HH_Sat_open = dummy[0][2]
            self.HH_Sat_close = dummy[0][3]
            self.HH_Sun_open = dummy[0][4]
            self.HH_Sun_close = dummy[0][5]
        except ValueError: 
            pass  
            
            
            
            
            
            
        conn.commit()    
        
        

     #get demands from time_start included to time_start excluded every HH. 
     #time_start and time-stop have to be supplied as seconds from the epoque divided 1800 (HH integer).    
    def getSimpleDemand(self,time_start,time_stop):
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        cur.execute('''SELECT Gas FROM Demand_Check Where Stores_id= ?''', (self.store_id,))
        dummy= cur.fetchone()
        try:
            if dummy[0] is not 1:
                raise  TypeError                 
            else:
                 cur.execute('''SELECT Time_id, Gas FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''', (self.store_id, time_start-1, time_stop))
                 RawData = cur.fetchall()
                 timeControl_start = RawData[0][0]
                 timeControl_stop = RawData[-1][0]
                 if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                     self.d_gas = np.array([elt[1] for elt in RawData])   
                     self.timestamp = np.array([elt[0] for elt in RawData])                
                 else:
                     print("time_id requested out of range. Gas range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                     raise ValueError                            
        except TypeError:
            print("We don't have the gas demand")
            
        cur.execute('''SELECT Ele FROM Demand_Check Where Stores_id= ?''', (self.store_id,))
        dummy= cur.fetchone()
        try:
            if dummy[0] is not 1:
                raise  TypeError                 
            else:
                 cur.execute('''SELECT Time_id, Ele FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''', (self.store_id, time_start-1, time_stop))
                 RawData = cur.fetchall()
                 timeControl_start = RawData[0][0]
                 timeControl_stop = RawData[-1][0]
                 if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                     self.d_ele = np.array([elt[1] for elt in RawData])                        
                 else:
                     print("time_id requested out of range. Electricity range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                     raise ValueError   
        except TypeError:
            print("We don't have the electricity demand")                 
        conn.commit() 
       
     #get prices from time_start included to time_start excluded every HH. 
     #time_start and time-stop have to be supplied as seconds from the epoque divided 1800 (HH integer).          
    def getSimplePrice(self, time_start, time_stop, string_table):           
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        try:
            sql_string = '''SELECT id, Ele, Gas, Ele_exp FROM {Table_name} Where DNO= ? AND Voltage = ? AND id > ? AND id < ?'''
            sql = sql_string.format(Table_name=string_table)
            cur.execute(sql, (self.DNO-9, self.Voltage, time_start-1, time_stop))        
            RawData = cur.fetchall()
            timeControl_start = RawData[0][0]
            timeControl_stop = RawData[-1][0]
            if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                 self.p_ele = np.array([elt[1] for elt in RawData]) 
                 self.p_gas = np.array([elt[2] for elt in RawData]) 
                 self.p_ele_exp = np.array([elt[3] for elt in RawData])  
                 self.timestamp = np.array([elt[0] for elt in RawData])                       
            else:
                 print("time_id requested out of range. Gas range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                 raise ValueError   
        except:
            print("An error occured. Possibly selected table doesn't exist. Please chose a valid Table")     
        conn.commit()
        
        

    def getTSODemand():
         pass