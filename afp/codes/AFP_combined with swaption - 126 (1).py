#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import scipy.stats as si
import math
import os

from scipy import interpolate
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.stats import norm


cur_path = '/Users/pranaykhattri/Downloads/UCLA_course_work/AFP/codeNData'

# Reading the data from excel for rates
data = pd.read_excel(cur_path+"/Dataset.xlsm", sheet_name='Rates', skiprows=3, usecols="A:L")
pd.to_datetime(data['Date'])

data = data.set_index('Date')
data.sort_index(inplace=True)
data.reset_index(inplace=True)


# In[2]:


# Obtaining given TBA profiles
data_TBA = pd.read_excel(cur_path+"/Dataset.xlsm", sheet_name='SampleTBA', skiprows=3, usecols="A:J")
data_Asset = pd.read_excel(cur_path+"/Dataset.xlsm", sheet_name='Asset', skiprows=5, usecols="A:F")

# data_TBA.values
data_TBA.set_index("Rate Shock (bps)", inplace=True)

# data_TBA.iloc[0,:]= data_TBA.iloc[0,:]*100
data_TBA

rate_change=[-200,-100,-50, -25, 0, 25, 50, 100, 200]
type(rate_change)
list(data_TBA.values)[0]

###TBA Profile Construction 
### Used for calculating TBA changes
Y=list(data_TBA.values)[0]
X=rate_change
X
Y

# Cubic spline fitting using TBA profile
from scipy.interpolate import CubicSpline 
cs_TBA = CubicSpline(X, Y)
Xcurve= np.arange(-400,400,.01)
cs_TBA(-300)

# cs_TBA(-300.25) # Check with decimals
#Look of TBA profile fitting with cubic spline
plt.plot(X, Y, 'o', Xcurve, cs_TBA(Xcurve), '-')
plt.legend(['data', 'cubic'], loc='best')
plt.show()


# In[3]:


### MSR profile fitting
data_Asset1= data_Asset.copy()
data_Asset1.drop(data_Asset1.index[13:23], inplace=True)

# Organizing data
data_Asset1.rename(columns={"Unnamed: 0": "Bps Change"}, inplace=True)

data_Asset1=data_Asset1.transpose() # run it 1 time as it will again flip
data_Asset1.index


# In[4]:


Bps_Changes=list(data_Asset1.loc["Bps Change"])
FNCL=list(data_Asset1.loc["FNCL"])
FNCI=list(data_Asset1.loc["FNCI"])
Five_yr=list(data_Asset1.loc["5 Yr"])
Seven_yr=list(data_Asset1.loc["7 yr"])
Ten_yr=list(data_Asset1.loc["10 Yr"])


# In[5]:


### Fitting Cubic Spline for each profile given to calculate the MSR changes
cs_FNCL = CubicSpline(Bps_Changes[0:13], FNCL[0:13])
cs_FNCI = CubicSpline(Bps_Changes[0:13], FNCI[0:13])
cs_Five_yr = CubicSpline(Bps_Changes[0:13], Five_yr[0:13])
cs_Seven_yr = CubicSpline(Bps_Changes[0:13], Seven_yr[0:13])
cs_Ten_yr = CubicSpline(Bps_Changes[0:13], Ten_yr[0:13])


# In[6]:


### Old work- TBA DV01 and CV01 calculated
bps_change=25
DV01_TBA=(data_TBA.values[0][3]-data_TBA.values[0][5])/(2*bps_change)

DV01_TBA

CV01_TBA=(data_TBA.values[0][3]+data_TBA.values[0][5])/(((bps_change)**2))
CV01_TBA ### negative convexity for TBA

### MSR values for given time frame- Taking out relevant rates column
MSR_Data= data[['Date','5 Yr', '7 Yr', '10 Yr','FNCL', 'FNCI']].copy()

#Converting into bps with daily changes
period=1
MSR_Data["5yr_Change"]= MSR_Data.loc[:,"5 Yr"].diff(periods=period)*100
MSR_Data["7yr_Change"]= MSR_Data.loc[:,"7 Yr"].diff(periods=period)*100
MSR_Data["10yr_Change"]= MSR_Data.loc[:,"10 Yr"].diff(periods=period)*100
MSR_Data["FNCL_Change"]= MSR_Data.loc[:,"FNCL"].diff(periods=period)*100
MSR_Data["FNCI_Change"]= MSR_Data.loc[:,"FNCI"].diff(periods=period)*100

MSR_Data

# Calculating the net MSR changes for respective rates using profile fittings
MSR_Data["MSR_Change_5yr"]= cs_Five_yr(MSR_Data["5yr_Change"])
MSR_Data["MSR_Change_7yr"]= cs_Seven_yr(MSR_Data["7yr_Change"])
MSR_Data["MSR_Change_10yr"]= cs_Ten_yr(MSR_Data["10yr_Change"])
MSR_Data["MSR_Change_FNCL"]= cs_FNCL(MSR_Data["FNCL_Change"])
MSR_Data["MSR_Change_FNCI"]= cs_FNCI(MSR_Data["FNCI_Change"])
MSR_Data

# Total MSR change
MSR_Data["MSR_Dollar_Change"]=MSR_Data["MSR_Change_5yr"] + MSR_Data["MSR_Change_7yr"]+ MSR_Data["MSR_Change_10yr"]+ MSR_Data["MSR_Change_FNCL"] + MSR_Data["MSR_Change_FNCI"]

MSR_Data # Comprising MSR value change with rates change

# Change in TBA value- using DV01, CV01- Dependent on FNCL- 30yr Mortgage Rate
MSR_Data["TBA_Dollar_Change"]= cs_TBA(MSR_Data["FNCL_Change"])

# cs_TBA(-6.67)
# MSR_Data.dtypes
MSR_Data

# Calculating MSR value on each day with changes and value given on 03/31/2020
MSR_Data_Valid_date= MSR_Data.copy()
MSR_Data_Valid_date.tail(16)

# Organizing data
MSR_Data_Valid_date.drop(MSR_Data_Valid_date.tail(15).index, inplace=True)

MSR_Data_Valid_date.sort_index(ascending=False,inplace=True)
MSR_Data_Valid_date

# MSR_Val=1098980779 #c Given value as on March 31-2020
MSR_Data_Valid_date["MSR_Value_After_Change"]= -MSR_Data_Valid_date["MSR_Dollar_Change"].shift(1)
MSR_Data_Valid_date

MSR_Data_Valid_date["MSR_Value_After_Change"].fillna(0, inplace=True)
MSR_Data_Valid_date

# MSR_Data_Valid_date.loc[3086,"MSR_Value_After_Change"]- MSR_Data_Valid_date.loc[3085,"MSR_Value_After_Change"] #Check:
MSR_Data_Valid_date["MSR_Value_After_Change"]= MSR_Data_Valid_date["MSR_Value_After_Change"].cumsum()
MSR_Data_Valid_date

MSR_Given_Value=1098980779 # Given value on March 31, 2020
MSR_Data_Valid_date

MSR_Data_Valid_date["MSR_Value_After_Change"]=MSR_Data_Valid_date["MSR_Value_After_Change"]+ MSR_Given_Value
MSR_Data_Valid_date

MSR_Data_Valid_date.sort_index(inplace=True)
MSR_Data_Valid_date

#MSR_Data_Valid_date.loc[1,"MSR_Value_After_Change"]-MSR_Data_Valid_date.loc[0,"MSR_Value_After_Change"]# Check
(MSR_Data_Valid_date[MSR_Data_Valid_date["MSR_Value_After_Change"]<0]).count().sum() # No negative value of MSR check

#31st March
data_Asset2= data_Asset.copy()
data_Asset2=data_Asset2.iloc[14:16,:]
MSR_DVO1_TBA=data_Asset2.loc[14,"FNCL"] + data_Asset2.loc[14,"FNCI"]
In_weights= -MSR_DVO1_TBA/DV01_TBA
In_weights

MSR_Data_Valid_date["Daily_TBA_Change"]=MSR_Data_Valid_date["TBA_Dollar_Change"]*In_weights
MSR_Data_Valid_date
#%%
##########################################################################


# In[62]:


#SWAPS:

#path_to_data = cur_path[0:cur_path.rindex('/')]+'/data'
data = pd.read_excel(open(cur_path+"/Dataset.xlsm", 'rb'), sheet_name='Rates', skiprows=3, usecols="A:J")
pd.to_datetime(data['Date'])
data = data.set_index('Date')
data.sort_index()


##Cubic Spline non-monotonic function - calculate swap rates at the defined points
def getSplineData(swap_rates, swap_tenors, x_points):
    cs = CubicSpline(swap_tenors, swap_rates)
    return cs(x_points)


## cleaning data - remove NAs
def cleanlist(swap_rates, tenors):
    clean_swap_rates = []
    clean_tenors = []
    for ind, rate in enumerate(swap_rates):
        if not math.isnan(rate):
            clean_swap_rates.append(rate)
            clean_tenors.append(tenors[ind])
    return clean_swap_rates, clean_tenors


## Bootstrap the discount factors 
# treat the swap rates as the par rates and bootstrap the discount factors starting from 0.5 maturity 
# upto the desired tenor. 
def bootstrapDt(swap_rates, tenors):
    dt = np.zeros(len(tenors))
    for i in np.arange(0, len(tenors)):
        dt[i] = (1 - sum(dt) * swap_rates[i] / 200) / (1 + swap_rates[i] / 200)
    return dt


def calcSwapKeyRateDV01(swap_rates, delta_y, tenors, key_rate_tenor, tenor_of_swap):
    # shift one swap rate by delta y to calculate Key Rate dv01 and cv01
    swap_rates_less = swap_rates.copy()

    # swap rate for swap we need to find the DV01 for:
    indx = tenors.tolist().index(tenor_of_swap, 0, len(tenors))

    # swap rate for which we are calculating the sensititvity
    indx_in_tnr = tenors.tolist().index(key_rate_tenor, 0, len(tenors))
    swap_rates_less[indx_in_tnr] = swap_rates_less[indx_in_tnr] - np.float64(delta_y)

    swap_rates_more = swap_rates.copy()
    swap_rates_more[indx_in_tnr] = swap_rates_more[indx_in_tnr] + np.float64(delta_y)

    discount_factors_less = bootstrapDt(swap_rates_less, tenors)
    discount_factors_more = bootstrapDt(swap_rates_more, tenors)
    swap_val_less = swap_rates[indx] / 2 * sum(discount_factors_less[:indx]) + discount_factors_less[indx]
    swap_val_more = swap_rates[indx] / 2 * sum(discount_factors_more[:indx]) + discount_factors_more[indx]
    key_rate_dv01 = (swap_val_less - swap_val_more) / (2 * delta_y)
    return (key_rate_dv01)


def calcSwapKeyRateCV01(swap_rates, delta_y, tenors, key_rate_tenor, tenor_of_swap):
    # shift one swap rate by delta y to calculate Key Rate dv01 and cv01
    swap_rates_less = swap_rates.copy()

    # swap rate for swap we need to find the DV01 for:
    indx = tenors.tolist().index(tenor_of_swap, 0, len(tenors))

    # swap rate for which we are calculating the sensititvity
    indx_in_tnr = tenors.tolist().index(key_rate_tenor, 0, len(tenors))
    swap_rates_less[indx_in_tnr] = swap_rates_less[indx_in_tnr] - np.float64(delta_y)

    swap_rates_more = swap_rates.copy()
    swap_rates_more[indx_in_tnr] = swap_rates_more[indx_in_tnr] + np.float64(delta_y)

    discount_factors_less = bootstrapDt(swap_rates_less, tenors)
    discount_factors_more = bootstrapDt(swap_rates_more, tenors)
    swap_val_less = swap_rates[indx] / 2 * sum(discount_factors_less[:indx]) + discount_factors_less[indx]
    swap_val_more = swap_rates[indx] / 2 * sum(discount_factors_more[:indx]) + discount_factors_more[indx]
    key_rate_cv01 = (swap_val_less + swap_val_more) / ((delta_y ** 2))
    return (key_rate_cv01)

# starting with the first day in the dataset "2007-09-27"
# we get all tenors for swaps

tenors = [1 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
currDate = "2020-03-31"
swap_rates = data.loc[data.index == currDate].values.tolist()[0]

# now we clean the data i.e. only keep the tenors for which a valid swap rate is available
y1, x1 = cleanlist(swap_rates, tenors)

# now fit the valid swap rates on the Cubic spline to interpolate 
swap_rates = getSplineData(y1, x1, np.arange(0.5, 10.5, 0.5))

# Calculating the discount factors
dt = bootstrapDt(swap_rates, np.arange(0.5, 10.5, 0.5))

dt
listOfSwaps = [5,7,10]

swapDV01=[]

# For each tenor, we calculate DV01 for notional $1
for swap in listOfSwaps:
    for swap_rate in listOfSwaps:
        print('Tenor', swap, ' and notional $1, Key rate DV01 for ', swap_rate, 'yr swap rate sensitivity is: ',
              calcSwapKeyRateDV01(swap_rates, .05, np.arange(0.5, (10 + 0.5), 0.5), swap_rate, swap))
        swapDV01.append(calcSwapKeyRateDV01(swap_rates, .05, np.arange(0.5, (10 + 0.5), 0.5), swap_rate, swap))
  
# calculating the changes in 5, 7 and 10 yr swaps by adding key rate changes w.r.t. change in 5, 7, 10 yr swap rates.        
MSR_Data_Valid_date.loc[:,'5yswapchanges']=MSR_Data_Valid_date.loc[:,'5yr_Change']*(swapDV01[0])+                                        (MSR_Data_Valid_date.loc[:,'7yr_Change']*swapDV01[1])+                                        (MSR_Data_Valid_date.loc[:,'10yr_Change']*swapDV01[2])
                                        
MSR_Data_Valid_date.loc[:,'7yswapchanges']=MSR_Data_Valid_date.loc[:,'5yr_Change']*(swapDV01[3])+                                        (MSR_Data_Valid_date.loc[:,'7yr_Change']*swapDV01[4])+                                        (MSR_Data_Valid_date.loc[:,'10yr_Change']*swapDV01[5])
                                        
MSR_Data_Valid_date.loc[:,'10yswapchanges']=MSR_Data_Valid_date.loc[:,'5yr_Change']*(swapDV01[6])+                                        (MSR_Data_Valid_date.loc[:,'7yr_Change']*swapDV01[7])+                                        (MSR_Data_Valid_date.loc[:,'10yr_Change']*swapDV01[8])
                                        
#DV01s From the dataset
MSR10YswapDV01=-2230976.25 
MSR7YswapDV01=-462444.19
MSR5YswapDV01=-760192.15
MSR10YswapCV01 = -37926
weights = np.zeros(7*3)
VaR_port = np.zeros(7)
Mean_port = np.zeros(7)


#SWAPTIONS:
##swaption price function
swaption = pd.read_excel(open(cur_path+"/Dataset.xlsm", 'rb'), sheet_name='Vols', skiprows=4, usecols="A:I")
swaption['Date'] = pd.to_datetime(swaption['Date'])
# swaption = swaption.set_index('Date')
# swaption.sort_index()
vol = swaption.loc[swaption['Date'] == currDate].values.tolist()[0]
del vol[0]
tenors = [1/12, 1, 2, 3, 5, 7, 10, 20, 30]
swap_rates = data.loc[data.index == currDate].values.tolist()[0]
y1, x1 = cleanlist(swap_rates, tenors)


def valueswaption(t1,t2,c,y1,updown,volinc=0):
    for i in range(len(y1)):
        y1[i] = y1[i]+updown
    if t1 == 0.25:
        swap_rates = getSplineData(y1, x1, np.arange(0.25,t2+t1+0.5,0.5))
        dt = bootstrapDt(swap_rates,  np.arange(0.25,t1+t2+0.5,0.5))
    else:
        swap_rates = getSplineData(y1, x1, np.arange(0.5,t2+t1+0.5,0.5))
        dt = bootstrapDt(swap_rates,  np.arange(0.5,t1+t2+0.5,0.5))
        tt=np.arange(0.5,t1+t2+0.5,0.5)
        tt=tt.tolist()
        t = tt.index(t1)
        dt=dt[t:]
    At = np.cumsum(dt)
    FSR = 2*((dt[0]-dt[len(dt)-1])/At[len(At)-1])
    if t1+t2 == 5.25:
        v = vol[0]
    elif t1+t2 == 5.5:
        v = vol[1]
    elif t1+t2 == 8:
        v = vol[2]
    elif t1+t2 == 10:
        v = vol[3]
    elif t1+t2 == 10.25:
        v = vol[4]
    elif t1+t2 == 10.5:
        v = vol[5]
    elif t1+t2 == 13:
        v = vol[6]
    elif t1+t2 == 15:
        v = vol[7]
    v = v+volinc
    d = (np.log(FSR/c)+((v**2)*t1)/2)/(v*(t1**0.5))
    price = -0.5*At[len(At)-1]*(FSR*norm.cdf(-d)-c*norm.cdf(-(d-v*(t1**0.5))))
    return price 

def calcSwaptionCV01(t1,t2,c,y1,delta_y,v):
    #shift curve by delta y to get dv01 and cv01
    swaption_less = valueswaption(t1,t2,c,y1,-delta_y,0)
    swaption_more = valueswaption(t1,t2,c,y1,delta_y,0)
    swaption_price = valueswaption(t1,t2,c,y1,0,0)
#     dv01 = (swaption_less-swaption_more)/(2*100*delta_y)
    cv01 = (swaption_less+swaption_more-2*swaption_price)
    return cv01


c=0.02
swaption_tenor = [(0.25,5),(0.5,5),(3,5),(5,5),(0.25,10),(0.5,10),(3,10),(5,10)]

##10yr - keyrate CV01 calculation
cv01 = [np.zeros(8)]
for i in range(len(swaption_tenor)):
    t1,t2 = swaption_tenor[i]
    bleh = calcSwaptionCV01(t1,t2,c,y1,.01,0)
    cv01.append(bleh)
cv01=cv01[1:8]

## swaption weight calculation
wt_swaption = -MSR10YswapCV01/max(cv01)
wt_swaption = int(wt_swaption)

## swaption VaR calculation
t1,t2 = swaption_tenor[cv01.index(max(cv01))]
x1
y1
##10yr - keyrate CV01 calculation
y1up = y1[:]
y1down = y1[:]
y1up[6] = y1up[6] + 0.01
y1down[6] = y1down[6] - 0.01
bleh = calcSwaptionCV01(t1,t2,c,y1up,0,0)+calcSwaptionCV01(t1,t2,c,y1down,0,0)-2*calcSwaptionCV01(t1,t2,c,y1,0,0)
keycv010=bleh

##7yr - keyrate CV01 calculation
y1up = y1[:]
y1down = y1[:]
y1up[5] = y1up[5] + 0.01
y1down[5] = y1down[5] - 0.01
bleh = calcSwaptionCV01(t1,t2,c,y1up,0,0)+calcSwaptionCV01(t1,t2,c,y1down,0,0)-2*calcSwaptionCV01(t1,t2,c,y1,0,0)
keycv07=bleh

##5yr - keyrate CV01 calculation
y1up = y1[:]
y1down = y1[:]
y1up[4] = y1up[4] + 0.01
y1down[4] = y1down[4] - 0.01
bleh = calcSwaptionCV01(t1,t2,c,y1up,0,0)+calcSwaptionCV01(t1,t2,c,y1down,0,0)-2*calcSwaptionCV01(t1,t2,c,y1,0,0)
keycv05=bleh


##keyrate DV01 calculation
keydv01 = [np.zeros(len(y1))]
for i in range(len(y1)):
    y1up = y1[:]
    y1down = y1[:]
    y1up[i] = y1up[i] + 0.01
    y1down[i] = y1down[i] - 0.01
    bleh = (valueswaption(t1,t2,c,y1down,0,0)-valueswaption(t1,t2,c,y1up,0,0))/2
    keydv01.append(bleh)

vega = (valueswaption(t1,t2,c,y1,0,1)-valueswaption(t1,t2,c,y1,0,-1))/2

if t1+t2 == 5.25:
    MSR_Data_Valid_date['vol_Change'] = swaption['3m5y'].diff(periods=period)
elif t1+t2 == 5.5:
    MSR_Data_Valid_date['vol_Change'] = swaption['5m5y'].diff(periods=period)
elif t1+t2 == 8:
    MSR_Data_Valid_date['vol_Change'] = swaption['3y5y'].diff(periods=period)
elif t1+t2 == 10:
    MSR_Data_Valid_date['vol_Change'] = swaption['6m5y'].diff(periods=period)
elif t1+t2 == 10.25:
    MSR_Data_Valid_date['vol_Change'] = swaption['3m10y'].diff(periods=period)
elif t1+t2 == 10.5:
    MSR_Data_Valid_date['vol_Change'] = swaption['6m10y'].diff(periods=period)
elif t1+t2 == 13:
    MSR_Data_Valid_date['vol_Change'] = swaption['3y10y'].diff(periods=period)
elif t1+t2 == 15:
    MSR_Data_Valid_date['vol_Change'] = swaption['5y10y'].diff(periods=period)
    
MSR10YswapDV01=MSR10YswapDV01+wt_swaption*keydv01[7]
MSR7YswapDV01=MSR7YswapDV01+wt_swaption*keydv01[6]
MSR5YswapDV01=MSR5YswapDV01+wt_swaption*keydv01[5]

MSR_Data_Valid_date.loc[:,'swaptionchanges']=-MSR_Data_Valid_date.loc[:,'5yr_Change']*(keydv01[5])-                                        (MSR_Data_Valid_date.loc[:,'7yr_Change']*keydv01[6])-                                        (MSR_Data_Valid_date.loc[:,'10yr_Change']*keydv01[7])+(0.5*(MSR_Data_Valid_date.loc[:,'10yr_Change']**2)*keycv010)+(0.5*(MSR_Data_Valid_date.loc[:,'7yr_Change']**2)*keycv07)+(0.5*(MSR_Data_Valid_date.loc[:,'5yr_Change']**2)*keycv05) +vega*MSR_Data_Valid_date.loc[:,'vol_Change']

# for i in range(2,len(MSR_Data_Valid_date)):
#     MSR_Data_Valid_date.loc[i,'swaptionchanges']=MSR_Data_Valid_date.loc[i-1,'swaptionchanges']-MSR_Data_Valid_date.loc[i,'5yr_Change']*(keydv01[5])-                  (MSR_Data_Valid_date.loc[i,'7yr_Change']*keydv01[6])-                                        (MSR_Data_Valid_date.loc[i,'10yr_Change']*keydv01[7])+0.5*(MSR_Data_Valid_date.loc[i,'10yr_Change']**2)*max(keycv01)+vega*MSR_Data_Valid_date.loc[i,'vol_Change']
# MSR_Data_Valid_date = MSR_Data_Valid_date[MSR_Data_Valid_date['Date']>"2010-03-31"]
# 0  1   2   3   4  5    6   7   8
# 55 57 510 75  77  710 105 107 1010

for i in range(7):
    if i==0:
        A = np.array([[swapDV01[0],swapDV01[3],swapDV01[6]],               [swapDV01[1],swapDV01[4],swapDV01[7]],               [swapDV01[2],swapDV01[5],swapDV01[8]]])
        B = np.array([-MSR5YswapDV01,-MSR7YswapDV01,-MSR10YswapDV01])
        C = np.linalg.solve(A,B)

        wt5yswap=C[0]
        wt7yswap=C[1]
        wt10yswap=C[2]
        
    if i==1:
        A = np.array([[swapDV01[0]+swapDV01[1]/2,swapDV01[6]+swapDV01[7]/2],               [swapDV01[2]+swapDV01[1]/2,swapDV01[8]+swapDV01[7]/2]])
        B = np.array([-MSR5YswapDV01-MSR7YswapDV01/2,-MSR7YswapDV01/2-MSR10YswapDV01])
        C = np.linalg.solve(A,B)
 
        wt5yswap=C[0]
        wt10yswap=C[1]
        wt7yswap=0
        
    if i==2:
        A = np.array([[swapDV01[4]+ swapDV01[3]/2,swapDV01[7]+ swapDV01[6]/2 ],               [swapDV01[5]+ swapDV01[3]/2,swapDV01[8]+ swapDV01[6]/2]])
        B = np.array([-MSR7YswapDV01-MSR5YswapDV01/2,-MSR10YswapDV01-MSR5YswapDV01/2])
        C = np.linalg.solve(A,B)
 
        wt7yswap=C[0]
        wt10yswap=C[1]
        wt5yswap=0
        
    if i==3:
        A = np.array([[swapDV01[0],swapDV01[3]],               [swapDV01[1],swapDV01[4]]])
        B = np.array([-MSR5YswapDV01-MSR10YswapDV01*0.5,-MSR7YswapDV01-MSR10YswapDV01*0.5])
        C = np.linalg.solve(A,B)
 
        wt5yswap=C[0]
        wt7yswap=C[1]
        wt10yswap=0
        
    if i==4:
        wt5yswap=(-MSR5YswapDV01-MSR7YswapDV01-MSR10YswapDV01)/(swapDV01[0]+swapDV01[1]+swapDV01[2])
        wt7yswap=0
        wt10yswap=0
        
    if i==5:
        wt5yswap=0
        wt7yswap=(-MSR5YswapDV01-MSR7YswapDV01-MSR10YswapDV01)/(swapDV01[3]+swapDV01[4]+swapDV01[5])
        wt10yswap=0
        
    if i==6:
        wt5yswap=0
        wt7yswap=0
        wt10yswap=(-MSR5YswapDV01-MSR7YswapDV01-MSR10YswapDV01)/(swapDV01[6]+swapDV01[7]+swapDV01[8])

    #
    MSR_Data_Valid_date.loc[:,'5Yrwt*changes']=-MSR_Data_Valid_date.loc[:,'5yswapchanges']*wt5yswap
    MSR_Data_Valid_date.loc[:,'7Yrwt*changes']=-MSR_Data_Valid_date.loc[:,'7yswapchanges']*wt7yswap
    MSR_Data_Valid_date.loc[:,'10Yrwt*changes']=-MSR_Data_Valid_date.loc[:,'10yswapchanges']*wt10yswap
    MSR_Data_Valid_date.loc[:,'swaptionwt*changes']=-MSR_Data_Valid_date.loc[:,'swaptionchanges']*wt_swaption

    wt5yswap,wt7yswap,wt10yswap
    #%%
#     MSR_Data_Valid_date.loc[:,'DailyPortfolioChanges']=MSR_Data_Valid_date.loc[:,'MSR_Dollar_Change']+MSR_Data_Valid_date.loc[:,'Daily_TBA_Change']+    MSR_Data_Valid_date.loc[:,'5Yrwt*changes']+MSR_Data_Valid_date.loc[:,'7Yrwt*changes']+MSR_Data_Valid_date.loc[:,'10Yrwt*changes']
    MSR_Data_Valid_date.loc[:,'DailyPortfolioChanges']=MSR_Data_Valid_date.loc[:,'MSR_Dollar_Change']+MSR_Data_Valid_date.loc[:,'Daily_TBA_Change']+    MSR_Data_Valid_date.loc[:,'5Yrwt*changes']+MSR_Data_Valid_date.loc[:,'7Yrwt*changes']+MSR_Data_Valid_date.loc[:,'10Yrwt*changes']+MSR_Data_Valid_date.loc[:,'swaptionwt*changes']

    #%%
    #VaR
    MSR_Data_Valid_date2=MSR_Data_Valid_date.copy()
    MSR_Data_Valid_date2.sort_values('DailyPortfolioChanges',inplace=True,ascending='True')
    VaR_99=MSR_Data_Valid_date2['DailyPortfolioChanges'].quantile(0.01,interpolation='lower')
    VaR_99
    mean_p=np.mean(MSR_Data_Valid_date2['DailyPortfolioChanges'])
    mean_p
    
    weights[i*3+0]=wt5yswap
    weights[i*3+1]=wt7yswap
    weights[i*3+2]=wt10yswap
    
    VaR_port[i] = VaR_99
    Mean_port[i] = mean_p
    MSR_Data_Valid_date2.iloc[np.where(MSR_Data_Valid_date2['DailyPortfolioChanges']==VaR_99)[0],:]
    #%%
    ##########################################################################


# In[86]:


#Min VaR port
i = VaR_port.tolist().index(max(VaR_port))
weights[i*3:(i+1)*3]

#Mean-VaR efficient port
Mean_VaR = Mean_port/VaR_port
Mean_VaR

n=[[5,7,10],[5,10],[7,10],[5,7],[5],[7],[10]]
fig, ax = plt.subplots()
ax.scatter((-VaR_port), Mean_port)

for i, txt in enumerate(n):
    ax.annotate(txt, (-VaR_port[i], Mean_port[i]))

plt.show()

i = Mean_VaR.tolist().index(max(Mean_VaR))
weights[i*3:(i+1)*3]


# In[84]:


Mean_VaR


plt.scatter((-VaR_port), Mean_port, marker='o')
# In[58]:


weights[i*3:(i+1)*3]
t1,t2

VaR_port

Mean_port
# df = pd.DataFrame((-VaR_port), Mean_port)
df = pd.DataFrame(MSR_Data_Valid_date2)
df.to_csv('pot.csv')
24,854,815.61069533