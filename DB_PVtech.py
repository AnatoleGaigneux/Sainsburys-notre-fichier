#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:13:44 2018

@author: Alex
"""

import sqlite3
import pandas as pd
import numpy as np


sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)    

conn = sqlite3.connect("Sainsburys.sqlite")
cur = conn.cursor()

#cur.execute('''SELECT * from Stores''')
#buba = cur.fetchall()

cur.executescript('''
DROP TABLE IF EXISTS PV_Technologies;

CREATE TABLE PV_Technologies (
    id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name   TEXT,
    price REAL NOT NULL, 
    Nominal_Power REAL NOT NULL, 
    Pv_eff REAL NOT NULL, 
    Capex REAL NOT NULL,
    Module_area REAL NOT NULL,
    Module_weight REAL NOT NULL,
    lifetime INTEGER NOT NULL
)

''')


bla1 = pd.read_excel('''PV_tech.xlsx''')
bla = bla1.iloc[0::,:]
for entry in range(len(bla)) :    
     entry_name  = bla.iloc[entry][0]; 
     entry_price = bla.iloc[entry][1];
     entry_Nominal_Power  = bla.iloc[entry][2];
     entry_Pv_eff  = bla.iloc[entry][3];
     entry_Capex  = bla.iloc[entry][4];
     entry_Module_area  = bla.iloc[entry][5];
     entry_Module_weight  = bla.iloc[entry][6];
     entry_lifetime  = bla.iloc[entry][7];
 

     #print entry_id
     cur.execute('''INSERT INTO PV_Technologies (name, price, Nominal_Power, Pv_eff, Capex, Module_area, Module_weight, lifetime) 
         VALUES ( ?,?,?,?,?,?,?,?)''', (entry_name, entry_price, entry_Nominal_Power, entry_Pv_eff, entry_Capex, entry_Module_area, entry_Module_weight, entry_lifetime) )   
conn.commit()    
     