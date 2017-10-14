# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:37:48 2017

@author: Ttle
"""


import sqlite3
#import Common.classStore as st
#import Common.classTech as tc
#import Solvers.classCHPProblem as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()


id_store_min =500
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

            cur.execute('''SELECT Ele FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''',
                        (store_index, time_start - 1, time_stop))
            Raw_data = cur.fetchall()
            Ele = np.array([elt[0] for elt in Raw_data])
            #Gas = np.array([elt[1] for elt in Raw_data])



            if Index[0] < 6 and Index[1] < 20000:
                category = 1 and Res_cat1.append([Ele])

            elif Index[0] < 6 and 20000 < Index[1] < 40000:
                category = 2 and Res_cat2.append([Ele])

            elif Index[0] < 6 and Index[1] > 40000:
                category = 3 and Res_cat3.append([Ele])

            elif Index[0] > 6 and Index[1] < 20000:
                category = 4 and Res_cat4.append([Ele])

            elif Index[0] > 6 and 20000 < Index[1] < 40000:
                category = 5 and Res_cat5.append([Ele])

            elif Index[0] > 6 and Index[1] > 40000:
                category = 6 and Res_cat6.append([Ele])

            else:
                print("Not able to categorise")




#calcualte averages across categories
#pd.average(Resutls_ele, )

#New = []
#New.append(Res_cat1)
#New.append(Res_cat2)
#New.append(Res_cat3)

Old = []
Old.append(Res_cat4)
#Old.append(Res_cat5)
#Old.append(Res_cat6)

print(np.transpose(Res_cat4))

#New_avg = np.average(np.transpose(New),axis=0)
Old_avg = np.average(np.transpose(Old),axis=0)
#Print("avg electricity demand of NEW stores is %s kW" % New_ele_avg)
print("avg electricity demand of OLD stores is %s kW" % Old_avg)

cat4 = np.array(Res_cat4)
cat4_avg = np.average(cat4, axis = 0)


plt.xlabel('')
plt.ylabel('Ele demand')
plt.plot(np.transpose(cat4_avg), 'ro', label = 'cat4')
plt.axis([0, len(Res_cat4), 0, 200])



legend = plt.legend(loc='lower left')
plt.show()
