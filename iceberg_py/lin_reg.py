#!/usr/bin/env pbkthon3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 23:04:08 2023

@author: laserglaciers
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

ak = np.array([11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]).reshape((-1,1))
bk = np.array([853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]).reshape((-1,1)) *-1
l_layer = np.arange(100,210,10).reshape((-1,1))
u_layer = np.arange(90,200,10).reshape((-1,1))

# model = LinearRegression()
model = LinearRegression().fit(ak, l_layer)
r_sq = model.score(ak, l_layer)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}\n")

# Make predictions
l_layer_test = model.predict(ak)

# plot lower labker
fig1, ax1 = plt.subplots()
ax1.scatter(ak,l_layer,color='k')
ax1.plot(ak,l_layer_test,color='b')
ax1.set_title('ak vs lower layer length')
ax1.set_ylabel('lower length')
ax1.set_xlabel('ak')



model = LinearRegression().fit(ak, l_layer)
r_sq = model.score(ak, l_layer)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}\n")

# Make predictions
mean = np.mean(np.diff(ak,axis=0))
ak2 = np.arange(ak[-1][0],-5,mean).reshape((-1,1))
l_layer_test = model.predict(ak2)

# plot lower labker
fig2, ax2 = plt.subplots()
ax2.scatter(ak,l_layer,color='k')
ax2.plot(ak2,l_layer_test,color='b')
ax2.set_title('ak vs lower layer length')
ax2.set_ylabel('lower length')
ax2.set_xlabel('ak')

step = 10
l_layer2 = np.arange(200,200+(len(ak2))*step,step).reshape((-1,1))


model3 = LinearRegression().fit(ak, bk)
r_sq3 = model3.score(ak, bk)
print(f"coefficient of determination: {r_sq3}")
print(f"intercept: {model3.intercept_}")
print(f"slope: {model3.coef_}\n")

# Make predictions
mean = np.mean(np.diff(ak,axis=0))
ak2 = np.arange(ak[-1][0],-5,mean).reshape((-1,1))
bk2 = model3.predict(ak2)

# plot lower labker
fig3, ax3 = plt.subplots()
ax3.scatter(ak,bk,color='k')
ax3.plot(ak2,bk2,color='b')
ax3.set_title('ak2 vs bk2')
ax3.set_ylabel('bk2')
ax3.set_xlabel('ak2')
ax3.text(6,-1000,f'{r_sq3:.2f}')









# # model = LinearRegression()
# model3 = LinearRegression().fit(bk, l_layer)
# r_sq3 = model.score(bk, l_layer)
# print(f"coefficient of determination: {r_sq3}")
# print(f"intercept: {model3.intercept_}")
# print(f"slope: {model3.coef_}\n")

# # Make predictions
# l_layer_test3 = model.predict(bk)

# # plot lower labker
# fig3, ax3 = plt.subplots()
# ax3.scatter(bk,l_layer, color='k')
# ax3.plot(bk,l_layer_test3, color='b')
# ax3.set_title('lower layer length vs bk')
