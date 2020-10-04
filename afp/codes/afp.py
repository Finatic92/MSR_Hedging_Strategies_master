
"""
This code takes input as the Dataset.xlsm and does the following:

1. reads the swap rates
2. fits the swap rates on cubic spline
3. bootstraps the discount factors and the spot rates
4. Values the At the Money - European Swaptions using Black Model
5. Calculates the DV01 and CV01 of Interest Rate Swaps using central difference
    approximations method for 1st and 2nd derivatives.
"""

# install xlrd
import pandas as pd
import numpy as np
from scipy import interpolate
import os
import scipy.stats as si
import math

cur_path = os.getcwd()
data = pd.read_excel(open("/Users/pranaykhattri/PycharmProjects/MSR_Hedging_Strategies_master/afp"+"/data/Dataset.xlsm", 'rb'), sheet_name='Rates', skiprows=3, usecols="A:J")
pd.to_datetime(data['Date'])

data = data.set_index('Date')
data.sort_index()

##Cubic Spline function
def getSplineData(swap_rates, swap_tenors, x_points):
    cs = interpolate.CubicSpline(swap_tenors, swap_rates)
    return cs(x_points)

## cleaning data
def cleanlist(swap_rates, tenors):
    clean_swap_rates = []
    clean_tenors = []
    for ind,rate in enumerate(swap_rates):
        if not math.isnan(rate):
            clean_swap_rates.append(rate)
            clean_tenors.append(tenors[ind])
    return clean_swap_rates, clean_tenors

tenors = [1/12, 1, 2, 3, 5, 7, 10, 20, 30]

##Bootstrap function
def bootstrapDt(swap_rates, tenors):
    dt = np.zeros(len(tenors))
    for i in np.arange(0,len(tenors)):
        dt[i] = (1-sum(dt)*swap_rates[i]/200)/(1+swap_rates[i]/200)
    return dt

##loop to get all the swap rates for all dates
dt = [np.arange(0,20)]
for i in np.arange(0,len(data.index)):
    swap_rates = data.loc[data.index == data.index[i]].values.tolist()[0]
    y1, x1 = cleanlist(swap_rates, tenors)
    swap_rates = getSplineData(y1, x1, np.arange(0.5,10.5,0.5))
    bleh = bootstrapDt(swap_rates,  np.arange(0.5,10.5,0.5))
    dt.append(bleh)

dt=pd.DataFrame(dt)
dt = dt.iloc[1:]
dt.index=data.index
dt.columns=np.arange(0.5,10.5,0.5)

## For 6m5year swaption
## Reading volatilities
vol = pd.read_excel(open("/Users/pranaykhattri/PycharmProjects/MSR_Hedging_Strategies_master/afp"+"/data/Dataset.xlsm", 'rb'), sheet_name='Vols', skiprows=4, usecols="A:J")
pd.to_datetime(vol['Date'])
vol = vol.set_index('Date')

##swaption price function
def swaptionprice(dt1,dt2,vol,t):
    price=(dt1-dt2)*(2*si.norm.cdf((math.sqrt(t*(vol)**2))/2, 0.0, 1.0)-1)
    return price

## Getting the discount factor values
dt1=dt[0.5]
dt1.index=dt.index
dt2=dt[5.5]
dt2.index=dt.index
prices=[]
dates=[]

## Looping through all the dates
for i in np.arange(0,len(vol.index)):
    dtt1= dt1.loc[dt1.index == vol.index[i]].values.tolist()[0]
    dtt2= dt2.loc[dt2.index == vol.index[i]].values.tolist()[0]
    bleh=swaptionprice(dtt1,dtt2,vol['6m5y'][i]/100,0.25)
    prices.append(bleh)

prices=pd.DataFrame(prices)
prices.index=vol.index

## pricing all swaptions for one given date

## Getting missing rates
dt = bootstrapDt(swap_rates,  np.arange(0.5,10.5,0.5))
R3m = getSplineData(y1, x1, 0.25)
R3m5y = getSplineData(y1, x1, 5.25)
R3m10y = getSplineData(y1, x1, 10.25)
rates2 = getSplineData(y1, x1, np.arange(0.5,15.5,0.5))

##calculating discount factors
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

##reading volatilities
swaption = pd.read_excel(open("/Users/pranaykhattri/PycharmProjects/MSR_Hedging_Strategies_master/afp"+"/data/Dataset.xlsm", 'rb'), sheet_name='Vols', skiprows=4, usecols="A:I")
pd.to_datetime(swaption['Date'])
swaption = swaption.set_index('Date')
swaption.sort_index()
vol = swaption.loc[swaption.index == "2007-09-27"].values.tolist()[0]
vol

##swaption pricing
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


# def getDiscountFactors(spots):
#     discount_factors = np.zeros(len(spots))
#     for i in np.arange(0,len(spots)):
#         discount_factors[i] = np.power(1/(1+spots[i]/2), (i+1))
#     return discount_factors



# Calculating DV01 and CV01 for IR Swaps
def calcSwapDV01(spots, swap_rate, delta_y, tenor):
    #shift curve by delta y to calculate dv01 and cv01
    discount_factors_less = getDiscountFactors(spots-delta_y)
    discount_factors_more = getDiscountFactors(spots+delta_y)
    swap_less = swap_rate/2 * sum(discount_factors_less) + discount_factors_less[-1]
    swap_more = swap_rate/2 * sum(discount_factors_more) + discount_factors_more[-1]
    dv01 = (swap_less-swap_more)/(2*delta_y)
    cv01 = (swap_less+swap_more-2)/(2*(delta_y**2))

