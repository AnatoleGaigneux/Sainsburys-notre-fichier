
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
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1])
                financials1.append(solution[4][4])
                carbon1.append(solution[5][2])
                store1.append(id_store)
                capex1.append(solution[4][5])
                biomethane1.append(solution[5][3])
                CHP1.append(solution[1])
                payback1.append(solution[4][1])

            if category == 2:
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1])
                financials2.append(solution[4][4])
                carbon2.append(solution[5][2])
                store2.append(id_store)
                capex2.append(solution[4][5])
                biomethane2.append(solution[5][3])
                CHP2.append(solution[1])
                payback2.append(solution[4][1])

            if category == 3:
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1])
                financials3.append(solution[4][4])
                carbon3.append(solution[5][2])
                store3.append(id_store)
                capex3.append(solution[4][5])
                biomethane3.append(solution[5][3])
                CHP3.append(solution[1])
                payback3.append(solution[4][1])


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
print('category1', CHP1)
print('category2', CHP2)
print('category3', CHP3)

average1 = np.average(carbon1)
average2 = np.average(carbon2)
average3 = np.average(carbon3)
print(average1, average2, average3)

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
print(Cat1_payback)
print(Cat2_payback)
print(Cat3_payback)


print(NumbStore1, NumbStore2, NumbStore3)

data_matrix = [['Category', 'Number of Stores', 'CAPEX (£)', 'BioMethane Quantity (GW/h)'],
               ['1', NumbStore1, cat1_capex, biometh_cat1],
               ['2', NumbStore2, cat2_capex, biometh_cat2],
               ['3', NumbStore3, cat3_capex, biometh_cat3],
               ['Overall Total', TotNumbStore, tot_capex, tot_biometh]]

table = ff.create_table(data_matrix)
py.offline.plot(table, filename='Results Summary table.html')

width = [average1, average2, average3]
cum_width = np.cumsum(width)
ind = [width[0]/2, (cum_width[1]-cum_width[0])/2+cum_width[0], (cum_width[2]-cum_width[1])/2+cum_width[1]]

plt.figure(3)
p1 = plt.bar(ind, MAC, width, linewidth=1, edgecolor='none')
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
