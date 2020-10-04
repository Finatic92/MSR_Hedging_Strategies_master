#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
from scipy import interpolate
import os

cur_path = os.getcwd()
data = pd.read_excel(open("/Users/nikhilaav/Documents/Coursework/AFP/Dataset.xlsm", 'rb'), sheet_name='Rates', skiprows=3, usecols="A:J")
pd.to_datetime(data['Date'])

data = data.set_index('Date')

data.sort_index()

def getSplineData(swap_rates, swap_tenors, x_points):
    tck = interpolate.splrep(swap_tenors, swap_rates)
    return interpolate.splev(x_points, tck)

def cleanlist(swap_rates, tenors):
    clean_swap_rates = []
    clean_tenors = []
    for ind,rate in enumerate(swap_rates):
        if rate != np.NaN:
            clean_swap_rates.append(rate)
            clean_tenors.append(tenors[ind])
    return clean_swap_rates, clean_tenors


tenors = [1/12, 1, 2, 3, 5, 7, 10, 20, 30]
currDate = "2007-09-27"
swap_rates = data.loc[data.index == "2007-09-27"].values.tolist()[0]

y1, x1 = cleanlist(swap_rates, tenors)
swap_rates = getSplineData(y1, x1, np.arange(0.5,10.5,0.5))

print(swap_rates,  np.arange(0.5,10.5,0.5))


def bootstrapDt(swap_rates, tenors):
    dt = np.zeros(len(tenors))
    for i in np.arange(0,len(tenors)):
        dt[i] = (1-sum(dt)*swap_rates[i]/200)/(1+swap_rates[i]/200)
    return dt

dt = bootstrapDt(swap_rates,  np.arange(0.5,10.5,0.5))

def getSpotRates(dt):
    spots = np.zeros(len(dt))
    for i in np.arange(0,len(dt)):
        spots[i] = (np.power(1/dt[i], 1/(i+1))-1)*2
    return spots

def getDiscountFactors(spots):
    discount_factors = np.zeros(len(spots))
    for i in np.arange(0,len(spots)):
        discount_factors[i] = np.power(1/(1+spots[i]/2), (i+1))
    return discount_factors

spots = getSpotRates(dt)

print(spots)

def calcSwapDV01(spots, swap_rate, delta_y, tenor):
    #shift curve by delta y to get dv01 and cv01
    discount_factors_less = getDiscountFactors(spots-delta_y)
    discount_factors_more = getDiscountFactors(spots+delta_y)
    swap_less = swap_rate/2 * sum(discount_factors_less) + discount_factors_less[-1]
    swap_more = swap_rate/2 * sum(discount_factors_more) + discount_factors_more[-1]
    dv01 = (swap_less-swap_more)/(2*delta_y)
    cv01 = (swap_less+swap_more-2)/(2*(delta_y**2))
    
print(dt)

R3m = getSplineData(y1, x1, 0.25)
R3m5y = getSplineData(y1, x1, 5.25)
R3m10y = getSplineData(y1, x1, 10.25)
rates2 = getSplineData(y1, x1, np.arange(0.5,15.5,0.5))

D3m = (1/(1+(R3m/100)))**(0.25)
D6m = dt[0]
D3y = dt[5]
D5y = dt[9]
D10y = dt[19]
D5y3m = (1-sum(dt[0:10])*R3m5y/200)
D5y6m = dt[10]
D5y3y = dt[15]
D5y5y = dt[19]
dt2 = bootstrapDt(rates2,  np.arange(0.5,15.5,0.5))
D10y3m = (1-sum(dt)*R3m10y/200)
D10y6m = dt2[20]
D10y3y = dt2[25]
D10y5y = dt2[29]

swaption = pd.read_excel(open("/Users/nikhilaav/Documents/Coursework/AFP/Dataset.xlsm", 'rb'), sheet_name='Vols', skiprows=4, usecols="A:I")
pd.to_datetime(swaption['Date'])
swaption = swaption.set_index('Date')
swaption.sort_index()
vol = swaption.loc[swaption.index == "2007-09-27"].values.tolist()[0]
vol

swaption_price = np.zeros(len(vol))
from scipy.stats import norm
swaption_price[0] = (D3m - D5y3m)*((2*norm.cdf((np.sqrt(((vol[0]/100)**2)*0.25))/2))-1)
swaption_price[1] = (D6m - D5y6m)*((2*norm.cdf((np.sqrt(((vol[1]/100)**2)*0.5))/2))-1)
swaption_price[2] = (D3y - D5y3y)*((2*norm.cdf((np.sqrt(((vol[2]/100)**2)*3))/2))-1)
swaption_price[3] = (D5y - D5y5y)*((2*norm.cdf((np.sqrt(((vol[3]/100)**2)*5))/2))-1)
swaption_price[4] = (D3m - D10y3m)*((2*norm.cdf((np.sqrt(((vol[4]/100)**2)*0.25))/2))-1)
swaption_price[5] = (D6m - D10y6m)*((2*norm.cdf((np.sqrt(((vol[5]/100)**2)*0.5))/2))-1)
swaption_price[6] = (D3y - D10y3y)*((2*norm.cdf((np.sqrt(((vol[6]/100)**2)*3))/2))-1)
swaption_price[7] = (D5y - D10y5y)*((2*norm.cdf((np.sqrt(((vol[7]/100)**2)*5))/2))-1)
swaption_price


# In[ ]:




