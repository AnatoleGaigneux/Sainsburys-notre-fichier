
import sqlite3
import Common.classStore as st
import Common.classTech as tc
import Solvers.classCHPProblemnew as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
import re

database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store_min = 0
id_store_max = 3000

time_start = 806448
time_stop = 824016

payback = []
h2p = []
Area = []
Ele_demand = []
Gas_demand = []
Age = []
carbon_savings = []
CHP_size = []
Capex = []
cum_disc_cashflow = []
financial_savings = []

for id_store in (j for j in range(id_store_min, id_store_max) if j != 2164):

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
            '''SELECT Area, GD2016, ED2016, Age FROM Stores Where id= {vn1}'''.format(
                vn1=id_store))
        Index = cur.fetchall()
        if not Index:
            pass
        else:
            SurfaceArea = np.array([elt[0] for elt in Index])
            Gas = np.array([elt[1] for elt in Index])
            Ele = np.array([elt[2] for elt in Index])
            Store_age = np.array([elt[3] for elt in Index])
            

            solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
            payback.append(solution[4][1])
            carbon_savings.append(solution[5][2])
            Capex.append(solution[4][5])
            cum_disc_cashflow.append(solution[4][4])
            CHP_size.extend(list(map(int, re.findall('\d+', solution[1]))))
            financial_savings.append(solution[4][0])
            h2p.append(Gas[0]/Ele[0])
            Area.append(SurfaceArea[0])
            Ele_demand.append(Ele[0])
            Gas_demand.append(Gas[0])
            Age.append(Store_age[0])
            
MAC = -1*np.array(cum_disc_cashflow)/abs(np.array(carbon_savings))

#Inputs =======================================================================
<<<<<<< HEAD
ind_variable = [Ele_demand, h2p] #possible independant variables: Ele_demand, Gas_demand, h2p, Area, Age
ind_var_name = ['Electricity demand (kW)','heat to power ratio']
=======
ind_variable = [np.divide(Ele_demand,1000000),h2p] #possible independant variables: Ele_demand, Gas_demand, h2p, Area, Age
ind_var_name = ['Electricity demand (GWh)','Heat to Power ratio']
init_guess = [1,1,1]  # 3 when 1 independant variable, 4 when 2 independant variables
<<<<<<< HEAD
=======
>>>>>>> 5fafcba68a5f615a7f6c96073fc2b8188ec0ae9f
init_guess = [1,0.0001,-5000]  # 3 when 1 independant variable, 4 when 2 independant variables
>>>>>>> ab7c22c72b825faafb527b172d703bb2175d58aa

dep_variable = carbon_savings #Possible dependant variables: payback, carbon_savings, CHP_size, Capex, financial_savings
dep_var_name = 'CHP size (MW)'
#==============================================================================

ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable = np.array(dep_variable, dtype=np.float64)

if len(ind_variable) == 1:
    ind_variable = ind_variable[0]
    def func(x, a, b): 
        return a*x+b
#    a*np.exp(-b*x)+c
    popt, pcov = curve_fit(func, ind_variable, dep_variable, init_guess)
    X = np.linspace(min(ind_variable), max(ind_variable),len(ind_variable))
    Y = func(X, *popt)
    plt.plot(X, Y, 'b-', label='fit' )
    plt.plot(ind_variable, dep_variable, 'ro', label='data')
    plt.xlabel(ind_var_name[0])
    plt.ylabel(dep_var_name)
    plt.legend()
    plt.show()

elif len(ind_variable) == 2:

    def func(x, a, b, c): 
        return a*x[0] + b*x[1] + c
    #a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
    popt, pcov = curve_fit(func, ind_variable, dep_variable, p0 = init_guess)
    
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.scatter(ind_variable[0], ind_variable[1], dep_variable, zdir='z', s=20, c='r', depthshade=True)
    x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
    x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
    x = [x1,x2]
    X1, X2 = np.meshgrid(x1, x2)
    zs = np.array([func([x1, x2],*popt) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = zs.reshape(X1.shape)
    ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
    ax.set_xlabel(ind_var_name[0], labelpad=8)
    ax.set_ylabel(ind_var_name[1], labelpad=8)
    ax.set_zlabel(dep_var_name,labelpad=8)
    ax.tick_params(axis='both', which='major', pad=3)
    plt.show()


#Calculate and print prediction error indicators
Target_test = dep_variable
Target_pred = func(ind_variable, *popt)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))


