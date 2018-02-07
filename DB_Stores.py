# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:36:07 2017

@author: nl211
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import date
import datetime


sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)  

conn = sqlite3.connect('Sainsburys.sqlite')
cur = conn.cursor()

cur.executescript('''
DROP TABLE IF EXISTS Stores;


CREATE TABLE Stores (
    id     INTEGER NOT NULL PRIMARY KEY UNIQUE,
    name   TEXT UNIQUE NOT NULL,
    DNO    INTEGER NOT NULL,
    SOD,
    Area,
    Age,
    ExistingTech,
    ElecEmissions,
    GasEmissions,
    Carbon,
    ED2016 FLOAT,
    GD2016 FLOAT,
    HH_WD_open,
    HH_WD_close,
    HH_Sat_open,
    HH_Sat_close,
    HH_Sun_open,
    HH_Sun_close
)

''')


#==============================================================================
# IMPORT STORES DATABASE
#==============================================================================

#bla = pd.read_csv('.\\RawData\\Stores.xlsx', names=['id','MPAN','DNO','name'])
#for entry in range(len(bla)) :    
#     entry_id = bla.iloc[entry][0];
#     entry_MPAN  = bla.iloc[entry][1];
#     entry_DNO  = bla.iloc[entry][2];
#     entry_name  = bla.iloc[entry][3];    
#     #print entry_id
#     cur.execute('''INSERT OR IGNORE INTO Stores (id,MPAN,DNO,name) 
#         VALUES ( ?,?,?,? )''', (entry_id,entry_MPAN,entry_DNO,entry_name ) )   
#conn.commit()    
#conn.close()
     

# 
#==============================================================================

df = pd.read_excel('Stores.xlsx', sheetname = 'Carbon Region Map',skiprows = 6)
for entry in range(len(df)) :    
     entry_id = df['StoreID'][entry]
     entry_DNO  = df['DNO ID'][entry]
     try:
        entry_Carbon= df['TC 2016-17'] [entry]
        entry_SOD = df['SOD (dd/mm/yy)'] [entry]
        entry_Area = df['Sales Area 16/17 (ft2)'][entry]
        entry_ElecEmissions = df['EC 2016-17'][entry]
        entry_GasEmissions = df['GC 2016-17'][entry]
        entry_ExistingTech = df['Notes'][entry]
     except:
         entry_Carbon= np.nan
         entry_SOD = np.nan
         entry_Area = np.nan
         entry_ElecEmissions=np.nan
         entry_GasEmissions=np.nan
         entry_ExistingTech = np.nan

     try:
         entry_Age = datetime.datetime.now().year-df['SOD (dd/mm/yy)'][entry].year
     except:
         entry_Age = np.nan

     entry_name  = df['Name'][entry]
     entry_ED2016  = df['ED 2016-17'][entry] 
     entry_GD2016 = df['GD 2016-17'][entry]
     try:
         entry_HH_WD_open = int(df['MON OPEN'][entry].hour*2 +  df['MON OPEN'][entry].minute/30)-1
         entry_HH_WD_close = int(df['MON CLOSE'][entry].hour*2 +  df['MON CLOSE'][entry].minute/30)-1
         entry_HH_Sat_open = int(df['SAT OPEN'][entry].hour*2  +  df['SAT OPEN'][entry].minute/30)-1
         entry_HH_Sat_close = int(df['SAT CLOSE'][entry].hour*2  +  df['SAT CLOSE'][entry].minute/30)-1
         entry_HH_Sun_open = int(df['SUN OPEN'][entry].hour*2  +  df['SUN OPEN'][entry].minute/30)-1
         entry_HH_Sun_close = int(df['SUN CLOSE'][entry].hour*2  +  df['SUN CLOSE'][entry].minute/30)-1
     except:
         entry_HH_WD_open = np.nan 
         entry_HH_WD_close = np.nan 
         entry_HH_Sat_open = np.nan 
         entry_HH_Sat_close = np.nan 
         entry_HH_Sun_open = np.nan 
         entry_HH_Sun_close = np.nan  
         
     #print entry_id
     try:
         cur.execute('''INSERT OR IGNORE INTO Stores (id,DNO,ElecEmissions,GasEmissions,Carbon,SOD,ExistingTech,Area,Age,name) 
             VALUES ( ?,?,?,?,?,?,?,?,?,?)''', (entry_id,entry_DNO,entry_ElecEmissions,entry_GasEmissions,entry_Carbon,entry_SOD,entry_ExistingTech,entry_Area, entry_Age,entry_name) )
     except:
         pass
     
     if   entry_ED2016 != 0:
        cur.execute('''UPDATE OR IGNORE Stores SET ED2016 = ? WHERE id = ?''', (entry_ED2016, entry_id  ) )  
     if   entry_GD2016 != 0:
        cur.execute('''UPDATE OR IGNORE Stores SET GD2016 = ? WHERE id = ?''', (entry_GD2016, entry_id  ) )      

     cur.execute('''UPDATE OR IGNORE Stores SET  HH_WD_open = ?,  HH_WD_close= ?,    HH_Sat_open= ?,    HH_Sat_close= ?,    HH_Sun_open= ?,   HH_Sun_close= ?  WHERE id = ? ''', ( entry_HH_WD_open,   entry_HH_WD_close,  entry_HH_Sat_open,  entry_HH_Sat_close,  entry_HH_Sun_open,  entry_HH_Sun_close, entry_id) )
 

conn.commit()    
conn.close()