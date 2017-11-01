
import sqlite3
import Common.classStore as st
import Common.classTech as tc
import Solvers.classCHPProblem as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

database_path = ".\\Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store_min = 0
id_store_max = 3000
store1, store2, store3 = ([] for i in range(3))
financials1, financials2, financials3 = ([] for i in range(3))
carbon1, carbon2, carbon3 = ([] for i in range(3))

for id_store in range(id_store_min, id_store_max ):
    goodIO = 0
    cur.execute('''SELECT Ele, Gas FROM Demand_Check Where Stores_id= {vn1}'''.format(vn1=id_store))
    checkIO = cur.fetchall()
    try:
        if checkIO[0][0] == 1:
            if checkIO[0][1] == 1:
                goodIO = 1
    except:
        pass

    if goodIO == 1:
        cur.execute(
            '''SELECT Area FROM Stores Where id= {vn1}'''.format(
                vn1=id_store))
        Index = cur.fetchall()
        if not Index:
            pass
        else:
            Area = np.array([elt[0] for elt in Index])

            if Area <= 25000:
                category = 1

            elif 25000 < Area <= 45000:
                category = 2

            elif Area > 45000:
                category = 3

            else:
                print("Not able to categorise")
            if category == 1:
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV()
                financials1.append(solution[4][4])
                carbon1.append(solution[5][2])
                store1.append(id_store)
            if category == 2:
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV()
                financials2.append(solution[4][4])
                carbon2.append(solution[5][2])
                store2.append(id_store)
            if category == 3:
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.2,1,1,1])
                financials3.append(solution[4][4])
                carbon3.append(solution[5][2])
                store3.append(id_store)

plt.figure(1)
plt.plot(store1, financials1, 'ro', label='<25000 $ft^2$')
plt.plot(store2, financials2, 'bo', label='25000-45000 $ft^2$')
plt.plot(store3, financials3, 'go', label='>45000 $ft^2$')
plt.xlabel('store id')
plt.ylabel('£ savings (cumul discounted cashflow)  2016-17')
plt.tight_layout()
plt.legend()

plt.figure(2)
plt.plot(store1, carbon1, 'ro', label='<25000 $ft^2$')
plt.plot(store2, carbon2, 'bo', label='25000-45000 $ft^2$')
plt.plot(store3, carbon3, 'go', label='>45000 $ft^2$')
plt.xlabel('store id')
plt.ylabel('carbon savings $tCO_2$ 2016-17')
plt.legend()
plt.tight_layout()
plt.show()
