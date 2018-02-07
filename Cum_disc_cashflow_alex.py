
import sqlite3
import Common.classStore as st
import Common.classTech as tc
import Solvers.classCHPProblemnew as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly as py
import plotly.figure_factory as ff

database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store_min = 0
id_store_max = 3000
store1, store2, store3 = ([] for i in range(3))
financials1, financials2, financials3 = ([] for i in range(3))
capex1, capex2, capex3 = ([] for i in range (3))
carbon1, carbon2, carbon3 = ([] for i in range(3))
biomethane1, biomethane2, biomethane3 = ([] for i in range(3))
CHP1, CHP2, CHP3 = ([] for i in range(3))
payback1, payback2, payback3 = ([] for i in range(3))
h2p = []



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
            '''SELECT Area, GD2016, ED2016 FROM Stores Where id= {vn1}'''.format(
                vn1=id_store))
        Index = cur.fetchall()
        if not Index:
            pass
        else:
            Area = np.array([elt[0] for elt in Index])
            Gas = np.array([elt[1] for elt in Index])
            Ele = np.array([elt[2] for elt in Index])

            if Area <= 25000:
                category = 1

            elif 25000 < Area <= 45000:
                category = 2

            elif Area > 45000:
                category = 3

            else:
                print("Not able to categorise")
            if category == 1:
                financials1.append(solution[4][4])
                carbon1.append(solution[5][2])
                store1.append(id_store)
                capex1.append(solution[4][5])
                biomethane1.append(solution[5][3])
                CHP1.append(solution[1])
                payback1.append(solution[4][1])

            if category == 2:
<<<<<<< HEAD
                #print(id_store)
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
=======
<<<<<<< HEAD
<<<<<<< HEAD
                #print(id_store)
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
=======
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [10.6/8.787,2.35/2.618,1,1])
>>>>>>> 7c6a9eeee8cc04bfd18fb2b7b4ea6a6d03a183d7
=======
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [10.6/8.787,2.35/2.618,1,1])
>>>>>>> ab7c22c72b825faafb527b172d703bb2175d58aa
>>>>>>> 5fafcba68a5f615a7f6c96073fc2b8188ec0ae9f
                financials2.append(solution[4][4])
                carbon2.append(solution[5][2])
                store2.append(id_store)
                capex2.append(solution[4][5])
                biomethane2.append(solution[5][3])
                CHP2.append(solution[1])
                payback2.append(solution[4][1])

            if category == 3:
<<<<<<< HEAD
                print(id_store)
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
=======
<<<<<<< HEAD
<<<<<<< HEAD
                print(id_store)
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
=======
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [10.6/8.787,2.35/2.618,1,1])
>>>>>>> 7c6a9eeee8cc04bfd18fb2b7b4ea6a6d03a183d7
=======
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [10.6/8.787,2.35/2.618,1,1])
>>>>>>> ab7c22c72b825faafb527b172d703bb2175d58aa
>>>>>>> 5fafcba68a5f615a7f6c96073fc2b8188ec0ae9f
                financials3.append(solution[4][4])
                carbon3.append(solution[5][2])
                store3.append(id_store)
                capex3.append(solution[4][5])
                biomethane3.append(solution[5][3])
                CHP3.append(solution[1])
                payback3.append(solution[4][1])
                h2p.append(Gas / Ele)

#print(-np.array(financials1)/abs(np.array(carbon1)))

print(-np.array(financials2)/abs(np.array(carbon2)))
'''plt.figure(1)
plt.plot(store1, financials1, 'ro', label='cat1')
plt.plot(store2, financials2, 'bo', label='cat2')
plt.plot(store3, financials3, 'go', label='cat3')
plt.legend()
plt.show()

plt.figure(2)

plt.plot(store1, carbon1, 'ro', label='cat1')
plt.plot(store2, carbon2, 'bo', label='cat2')
plt.plot(store3, carbon3, 'go', label='cat3')
plt.legend()
plt.show()'''

MAC = [-np.average(financials1)/abs(np.average(carbon1)),
       -np.average(financials2)/abs(np.average(carbon2)),
       -np.average(financials3)/abs(np.average(carbon3))]
#print('category1', CHP1)
#print('category2', CHP2)
#print('category3', CHP3)

average1 = np.average(carbon1)
average2 = np.average(carbon2)
average3 = np.average(carbon3)
#print(average1, average2, average3)

# Capex Calculation
cat1_capex = np.sum(capex1)
cat2_capex = np.sum(capex2)
cat3_capex = np.sum(capex3)
tot_capex = cat1_capex + cat2_capex + cat3_capex

# Biomethane Quantity Calculation
biometh_cat1 = np.sum(biomethane1)/10**6
biometh_cat2 = np.sum(biomethane2)/10**6
biometh_cat3 = np.sum(biomethane3)/10**6
tot_biometh = biometh_cat1 + biometh_cat2 + biometh_cat3

# Number of store per category
NumbStore1 = len(store1)
NumbStore2 = len(store2)
NumbStore3 = len(store3)
TotNumbStore = NumbStore1 + NumbStore2 + NumbStore3

# Payback time
Cat1_payback=np.average(payback1)
Cat2_payback=np.average(payback2)
Cat3_payback=np.average(payback3)
#print(Cat1_payback)
#print(Cat2_payback)
#print(Cat3_payback)


#print(NumbStore1, NumbStore2, NumbStore3)

data_matrix = [['Category', 'Number of Stores', 'CAPEX (£)', 'BioMethane Quantity (GWh)'],
               ['1', NumbStore1, cat1_capex, biometh_cat1],
               ['2', NumbStore2, cat2_capex, biometh_cat2],
               ['3', NumbStore3, cat3_capex, biometh_cat3],
               ['Overall Total', TotNumbStore, tot_capex, tot_biometh]]

table = ff.create_table(data_matrix)
py.offline.plot(table, filename='Results Summary table.html')

Bigstores=np.transpose(store3)
Payback3=np.transpose(payback3)
CAPEX=np.transpose(capex3)
Heat2Power=np.transpose(h2p)
CHP=np.transpose(CHP3)
Financial=np.transpose(-np.array(financials3)/abs(np.array(carbon3)))
C2C=np.transpose(np.array(carbon3)*1000/np.array(capex3))

#print(Bigstores)
#print(Payback3)
#print(CAPEX)

data_matrix = [['Store ID', 'Payback', 'CAPEX (£)','Heat to Power Ratio','CHP','Financial Benefit(£/tCo2)','Carbon Savings/Capex kgCO2/£'],
               [Bigstores.item(0), Payback3.item(0), CAPEX.item(0),Heat2Power.item(0),CHP.item(0),Financial.item(0),C2C.item(0)],
               [Bigstores.item(1), Payback3.item(1), CAPEX.item(1),Heat2Power.item(1),CHP.item(1),Financial.item(1),C2C.item(1)],
               [Bigstores.item(2), Payback3.item(2), CAPEX.item(2),Heat2Power.item(2),CHP.item(2),Financial.item(2),C2C.item(2)],
               [Bigstores.item(3), Payback3.item(3), CAPEX.item(3),Heat2Power.item(3),CHP.item(3),Financial.item(3),C2C.item(3)],
               [Bigstores.item(4), Payback3.item(4), CAPEX.item(4),Heat2Power.item(4),CHP.item(4),Financial.item(4),C2C.item(4)],
               [Bigstores.item(5), Payback3.item(5), CAPEX.item(5),Heat2Power.item(5),CHP.item(5),Financial.item(5),C2C.item(5)],
               [Bigstores.item(6), Payback3.item(6), CAPEX.item(6),Heat2Power.item(6),CHP.item(6),Financial.item(6),C2C.item(6)],
               [Bigstores.item(7), Payback3.item(7), CAPEX.item(7),Heat2Power.item(7),CHP.item(7),Financial.item(7),C2C.item(7)],
               [Bigstores.item(8), Payback3.item(8), CAPEX.item(8),Heat2Power.item(8),CHP.item(8),Financial.item(8),C2C.item(8)],
               [Bigstores.item(9), Payback3.item(9), CAPEX.item(9),Heat2Power.item(9),CHP.item(9),Financial.item(9),C2C.item(9)],
               [Bigstores.item(10), Payback3.item(10), CAPEX.item(10),Heat2Power.item(10),CHP.item(10),Financial.item(10),C2C.item(10)],
               [Bigstores.item(11), Payback3.item(11), CAPEX.item(11),Heat2Power.item(11),CHP.item(11),Financial.item(11),C2C.item(11)],
               [Bigstores.item(12), Payback3.item(12), CAPEX.item(12),Heat2Power.item(12),CHP.item(12),Financial.item(12),C2C.item(12)],
               [Bigstores.item(13), Payback3.item(13), CAPEX.item(13),Heat2Power.item(13),CHP.item(13),Financial.item(13),C2C.item(13)],
               [Bigstores.item(14), Payback3.item(14), CAPEX.item(14),Heat2Power.item(14),CHP.item(14),Financial.item(14),C2C.item(14)],
               [Bigstores.item(15), Payback3.item(15), CAPEX.item(15),Heat2Power.item(15),CHP.item(15),Financial.item(15),C2C.item(15)],
               [Bigstores.item(16), Payback3.item(16), CAPEX.item(16),Heat2Power.item(16),CHP.item(16),Financial.item(16),C2C.item(16)],
               [Bigstores.item(17), Payback3.item(17), CAPEX.item(17),Heat2Power.item(17),CHP.item(17),Financial.item(17),C2C.item(17)]]

table = ff.create_table(data_matrix)
py.offline.plot(table, filename='Category 3 breakdown.html')


#print('Store Id',store3)
#print('Payback',payback3)
#print('H2P', h2p)
#print('CHP', CHP3)

width = [average1, average2, average3]
cum_width = np.cumsum(width)
ind = [width[0]/2, (cum_width[1]-cum_width[0])/2+cum_width[0], (cum_width[2]-cum_width[1])/2+cum_width[1]]

plt.figure(3)
plt.xlabel('$tCO_2e$ yearly savings')
plt.ylabel('$£/tCO_2e$')
plt.title('MAC curves for each store category and CHP implementation 2016-17')
plt.show()




# rects = p1.patches
# labels = ['Category 1', 'Category 2', 'Category 3']
# for rect, label in zip(rects, labels):
#    height = rect.get_height()
#    p1.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
# plt.show()
