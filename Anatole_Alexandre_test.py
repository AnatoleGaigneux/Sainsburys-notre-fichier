# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:37:48 2017

@author: Ttle
"""


import sqlite3
import Common.classStore as st
import Common.classTech as tc
#import Solvers.classCHPProblem as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hello
database_path = ".\\Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()


id_store_min = 2000
id_store_max =2050

time_start = 806448
time_stop = 824016

Res_cat1 = []
Res_cat2 = []
Res_cat3 = []
Res_cat4 = []
Res_cat5 = []
Res_cat6 = []

for store_index in range(id_store_min, id_store_max ): 
        
#store_index = Anatole's stores
        
        goodIO = 0
        cur.execute('''SELECT Ele, Gas FROM Demand_Check Where Stores_id= {vn1}'''.format(vn1 = store_index))
        checkIO = cur.fetchall()
        try:
           if checkIO[0][0] == 1:
               if checkIO[0][1] == 1:
                    goodIO = 1
        except:
            pass

        if goodIO == 1:
            '''
            categorise the store
            Categories:
            1 new+small
            2 new + medium
            3 new + large
            4 old + small
            5 old + medium
            6 old + large
            '''

            cur.execute('''SELECT Age, Area FROM Stores Where id= {vn1}'''.format(vn1=store_index))
            Index = cur.fetchone()
            if Index[0] < 6 and Index[1] < 20000:
                category = 1
            elif Index[0] < 6 and 20000 < Index[1] < 40000:
                category = 2
            elif Index[0] < 6 and Index[1] > 40000:
                category = 3
            elif Index[0] > 6 and Index[1] < 20000:
                category = 4
            elif Index[0] > 6 and 20000 < Index[1] < 40000:
                category = 5
            elif Index[0] > 6 and Index[1] > 40000:
                category = 6
            else:
                print("Not able to categorise")


            # get demand
            cur.execute('''SELECT Ele, Gas FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''', (store_index, time_start-1, time_stop))
            Raw_data = cur.fetchall()
            Ele = np.array([elt[0] for elt in Raw_data])
            Gas = np.array([elt[1] for elt in Raw_data])
            
            if category == 1:
                Res_cat1.append([Ele])
            elif category == 2:
                Res_cat2.append([Ele])
            elif category == 3:
                Res_cat3.append([Ele])
            elif category == 4:
                Res_cat4.append([Ele])
            elif category == 5:
                Res_cat5.append([Ele])
            elif category == 6:
                Res_cat6.append([Ele])
            else:
                print("Data not added in matrix")
#calcualte averages across categories
#pd.average(Results_ele, )
        break
New = []
New.append(Res_cat1)
New.append(Res_cat2)
New.append(Res_cat3)

Old = []
Old.append(Res_cat4)
Old.append(Res_cat5)
Old.append(Res_cat6)

print(Res_cat5)

"""
New_avg = np.average(New, axis=0)
Old_avg = np.average(New, axis=0)
Print("avg electricity demand of NEW stores is %s kW" % New_ele_avg)
Print("avg electricity demand of OLD stores is %s kW" % Old_ele_avg)
'''
# cat1 = np.array(Res_cat1)
# cat1_avg = np.average(cat1, axis = 0)


plt.xlabel('')
plt.ylabel('Ele demand')
plt.axis([0, 300, 0, 200])
plt.plot(cat1_avg, 'ro', label = 'cat1')


legend = plt.legend(loc='lower left')
plt.show()
"""