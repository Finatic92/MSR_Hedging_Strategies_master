# Created by pranaykhattri at 19/09/20
# install xlrd
import pandas as pd
import numpy as np
import datetime
from scipy import interpolate
import os

cur_path = os.getcwd()
data = pd.read_excel(open("/Users/pranaykhattri/PycharmProjects/MSR_Hedging_Strategies_master/afp"+"/data/Dataset.xlsm", 'rb'), sheet_name='Rates', skiprows=3, usecols="A:J")
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
    dv01 = (swap_more-swap_less)/(2*delta_y)

