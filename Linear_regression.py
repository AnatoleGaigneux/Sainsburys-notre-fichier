
import sqlite3
import Common.classStore as st
import Common.classTech as tc
import Solvers.classCHPProblemnew as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D
import plotly as py
import plotly.figure_factory as ff
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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
            SurfaceArea = np.array([elt[0] for elt in Index])
            Gas = np.array([elt[1] for elt in Index])
            Ele = np.array([elt[2] for elt in Index])
            
            if SurfaceArea > 45000:
                print(id_store)
                solution = pb.CHPproblem(id_store).SimpleOpti5NPV(mod = [1.195,1,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
                payback.append(solution[4][1])
                h2p.append(Gas/Ele)
                Area.append(SurfaceArea)

print(h2p)
#Split data into training/testing sets

X1_train = Area[:-9]
X1_test = Area[-9:]

X2_train = h2p[:-9]
X2_test = h2p[-9:]

X_train = np.column_stack((X1_train,X2_train))
X_test = np.column_stack((X1_test,X2_test))

#Split targets into trainign/testing sets
target_train = payback[:-9]
target_test = payback[-9:]
           
            
#fit a linear model

lm = linear_model.LinearRegression()
model = lm.fit(X1_train, target_train)
            
#Make predictions using the testing sets
            
target_pred = lm.predict(X_test)

#get coefficents
print('Coefficients: \n', lm.coef_)
#print mean squared error
print("Mean squared error: %.2f" % mean_squared_error(target_test, target_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(target_test, target_pred))

#Plot outputs

fig = plt.figure(1)
ax = Axes3D(fig)
ax.scatter(X1_test, X2_test, target_test, zdir='z', s=20, c='r', depthshade=True)

X1_range = np.arange(min(Area),max(Area))
X2_range =  np.arange(min(h2p),max(h2p))
params = lm.coef_
Y = X1_range*params[0]+ X2_range*params[1]+lm.intercept_
ax.plot_surface(X1_range, X2_range, Y, cmap='binary', linewidth=0, antialiased=False)
ax.set_xlabel('Area (ft2)')
ax.set_ylabel('Heat to power')
ax.set_zlabel('Payback time')
ax.tick_params(axis='both', which='major', pad=-5)

plt.figure(2)
plt.plot(X2_train, 'ro')
plt.plot(X2_test, 'bo')

plt.show()
