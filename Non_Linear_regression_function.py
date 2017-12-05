
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
Store = []
id_store_min = 0
id_store_max = 3000
time_start = 806448
time_stop = 824016

elec_price=11.9

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
            
            Store.append(id_store)
            solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [elec_price/8.787,2.35/2.618,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
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
    
#Inputs =======================================================================
ind_variable = [np.divide(Ele_demand,1000000),h2p] #possible independant variables: Ele_demand, Gas_demand, h2p, Area, Age
ind_var_name = ['Electricity demand (GWh)','Heat to Power ratio']

dep_variable1 = payback #Possible dependant variables: payback, carbon_savings, CHP_size, Capex, financial_savings, cum_disc_cashflow
dep_var_name1 = 'Payback time (Years)'
dep_variable2 = carbon_savings
dep_var_name2 = 'Carbon reduction (tCO2)'
dep_variable3 = np.divide(financial_savings,1000)
dep_var_name3 = 'Operating cost reduction (Thousands £)'
dep_variable4 = np.divide(cum_disc_cashflow,1000)
dep_var_name4 = 'Cumulative Discounted Cash Flow \n (Thousand £)'
dep_variable5 = CHP_size
dep_var_name5 = 'CHP size (MW)'
#==============================================================================
# ============================================================================
#PAYBACK
# ============================================================================
ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable1 = np.array(dep_variable1, dtype=np.float64)
    
def func1(x, a, b, c, d): 
    return a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
#a*x[0] + b*x[1] + c
popt1, pcov1 = curve_fit(func1, ind_variable, dep_variable1)

fig = plt.figure(1)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable1, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func1([x1, x2],*popt1) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel(ind_var_name[0], labelpad=8)
ax.set_ylabel(ind_var_name[1], labelpad=8)
ax.set_zlabel(dep_var_name1,labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()
    
        
#Calculate and print prediction error indicators
Target_test = dep_variable1
Target_pred = func1(ind_variable, *popt1)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
print('-------payback')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))

# ============================================================================
#CARBON SAVINGS
# ============================================================================
ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable2 = np.array(dep_variable2, dtype=np.float64)
        
def func2(x, a, b, c): 
    return a*x[0] + b*x[1] + c
popt2, pcov2 = curve_fit(func2, ind_variable, dep_variable2)

fig = plt.figure(2)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable2, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func2([x1, x2],*popt2) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel(ind_var_name[0], labelpad=8)
ax.set_ylabel(ind_var_name[1], labelpad=8)
ax.set_zlabel(dep_var_name2,labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()
    
        
#Calculate and print prediction error indicators
Target_test = dep_variable2
Target_pred = func2(ind_variable, *popt2)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
print('-------Carbon savings')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))

# ============================================================================
#FINANCIAL SAVINGS
# ============================================================================
ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable3 = np.array(dep_variable3, dtype=np.float64)
        
def func3(x, a, b, c): 
    return a*x[0] + b*x[1] + c
popt3, pcov3 = curve_fit(func3, ind_variable, dep_variable3)

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable3, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func3([x1, x2],*popt3) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel(ind_var_name[0], labelpad=8)
ax.set_ylabel(ind_var_name[1], labelpad=8)
ax.set_zlabel(dep_var_name3,labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()
    
        
#Calculate and print prediction error indicators
Target_test = dep_variable3
Target_pred = func3(ind_variable, *popt3)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
print('------Financial savings')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))

# ============================================================================
#CUMULATIVE DISCOUNTED CASHFLOW
# ============================================================================
ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable4 = np.array(dep_variable4, dtype=np.float64)
        
def func4(x, a, b, c): 
    return a*x[0] + b*x[1] + c
popt4, pcov4 = curve_fit(func4, ind_variable, dep_variable4)

fig = plt.figure(4)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable4, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func4([x1, x2],*popt4) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel(ind_var_name[0], labelpad=8)
ax.set_ylabel(ind_var_name[1], labelpad=8)
ax.set_zlabel(dep_var_name3,labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()
    
        
#Calculate and print prediction error indicators
Target_test = dep_variable4
Target_pred = func4(ind_variable, *popt4)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
print('------Cumulative discounted cash flow')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))

# ============================================================================
#CHPsize
# ============================================================================
ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable5 = np.array(dep_variable5, dtype=np.float64)      

def func5(x, a, b): 
    return a*x+b
    #   a*np.exp(-b*x)+c
popt5, pcov5 = curve_fit(func5, ind_variable[0], dep_variable5)
X = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
Y = func5(X, *popt5)
plt.figure(5)
plt.plot(X, Y, 'b-', label='fit' )
plt.plot(ind_variable[0], dep_variable5, 'ro', label='data')
plt.xlabel(ind_var_name[0])
plt.ylabel(dep_var_name3)
plt.legend()
plt.show()
    
#Calculate and print prediction error indicators
Target_test = dep_variable5
Target_pred = func5(ind_variable[0], *popt5)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/abs(Target_pred[i]))*100)
print('------CHP size')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))