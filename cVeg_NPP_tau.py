import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pymannkendall as mk
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,7))

fig.subplots_adjust(hspace=0.22)
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.08)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.93)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1=fig.add_subplot(2,3,1)
ax2=fig.add_subplot(2,3,2)
ax3=fig.add_subplot(2,3,3)
ax4=fig.add_subplot(2,3,4)
ax5=fig.add_subplot(2,3,5)
ax6=fig.add_subplot(2,3,6)

GCMs = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'EC-Earth3', 'EC-Earth3-Veg',
        'GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
        'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
        'NorESM2-LM', 'NorESM2-MM']

GCMs_CMIP = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM','EC-Earth3-Veg',
             'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MPI-ESM1-2-LR',
             'NorESM2-LM', 'NorESM2-MM']

def read_in(file,var,experiment,GCM,constraint,fert,LPJ):
    if LPJ == False:
        fname = ('../../CMIP6/'+var+'_ssp'+experiment+'/'+GCM+'/'+var+'_day_'+
                GCM+'_ssp'+experiment+'_r1i1p1f1_gn_1850-2100_oz.nc')
    elif LPJ == True:
        if constraint == True:
            if fert == True:
                fname=('../SSP'+experiment+'_QM/'+GCM+'/'+file+'_'+GCM+
                       '_1850-2100_oz.nc')
            else:
                fname=('../SSP'+experiment+'_QM/'+GCM+'_SE_CO2/'+file+'_'+GCM+
                       '_1850-2100_oz.nc')
        elif constraint == False:
            fname=('../SSP'+experiment+'_raw/'+GCM+'/'+file+'_'+GCM+
                   '_1850-2100_oz.nc')

    ds = xr.open_dataset(fname)
    return(ds[var].to_numpy().flatten())

def tau(experiment,GCM,constraint,fert,LPJ):
    if LPJ == True:
        file_npp = 'anpp'
        var_npp = 'Total'
        file_cveg = 'cpool'
        var_cveg = 'VegC'
    else:
        file_npp = 'npp'
        var_npp = 'npp'
        file_cveg = 'cVeg'
        var_cveg = 'cVeg'

    NPP = read_in(file_npp,var_npp,experiment,GCM,constraint,fert,LPJ)
    CVeg = read_in(file_cveg,var_cveg,experiment,GCM,constraint,fert,LPJ)

    NPP = NPP[51:]
    CVeg = CVeg[51:]

    time = np.arange(1901,2101)

    tau = []

    for i in range(0,199):
        tau_prel = CVeg[i+1]/ (NPP[i+1] - ((CVeg[i+1] - CVeg[i])/ (time[i+1] -
                   time[i])))
        tau.append(tau_prel)
        
    return(tau)

df_LPJ_cVeg_raw = pd.DataFrame()
df_LPJ_cVeg_QM = pd.DataFrame()
df_LPJ_cVeg_QM_CO2 = pd.DataFrame()
df_CMIP_cVeg = pd.DataFrame()

df_LPJ_NPP_raw = pd.DataFrame()
df_LPJ_NPP_QM = pd.DataFrame()
df_LPJ_NPP_QM_CO2 = pd.DataFrame()
df_CMIP_NPP = pd.DataFrame()

df_LPJ_tau_raw = pd.DataFrame()
df_LPJ_tau_QM = pd.DataFrame()
df_LPJ_tau_QM_CO2 = pd.DataFrame()
df_CMIP_tau = pd.DataFrame()

for m in GCMs:
    df_LPJ_cVeg_raw[m] = read_in('cpool','VegC','585',m,False,'',True)
    df_LPJ_cVeg_QM[m] = read_in('cpool','VegC','585',m,True,True,True)
    df_LPJ_NPP_raw[m] = read_in('anpp','Total','585',m,False,'',True)
    df_LPJ_NPP_QM[m] = read_in('anpp','Total','585',m,True,True,True)
    df_LPJ_tau_raw[m]=tau('585',m,False,'',True)
    df_LPJ_tau_QM[m]=tau('585',m,True,True,True)

for m in GCMs:
    df_LPJ_cVeg_QM_CO2[m] = read_in('cpool','VegC','585',m,True,False,True)
    df_LPJ_NPP_QM_CO2[m] = read_in('anpp','Total','585',m,True,False,True)
    df_LPJ_tau_QM_CO2[m]=tau('585',m,True,False,True)

for m in GCMs_CMIP:
    df_CMIP_cVeg[m]=read_in('cVeg','cVeg','585',m,'','',False)
    df_CMIP_NPP[m]=read_in('npp','npp','585',m,'','',False)
    df_CMIP_tau[m]=tau('585',m,False,'',False)

def plot(dataset,axis,color,ls):
    df_change = dataset.rolling(window=10,center=True).mean()
    df_stats = pd.DataFrame()
    df_stats['median'] = df_change.median(axis=1)
    df_stats['min'] = df_change.min(axis=1)
    df_stats['max'] = df_change.max(axis=1)

    if axis in (ax3, ax6):
        df_stats['Year'] = np.arange(1902,2101)
    else:
        df_stats['Year'] = np.arange(1850,2101)

    axis.fill_between(df_stats['Year'],df_stats['min'],df_stats['max'],
                      color=color)
    axis.plot(df_stats['Year'],df_stats['median'],color='k',ls=ls)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.set_xlim(1901,2100)

plot(df_LPJ_cVeg_raw,ax2,'#b2d6ea','-')
plot(df_LPJ_cVeg_QM,ax2,'#3292c8',':')
plot(df_LPJ_cVeg_QM_CO2,ax2,'#005f95','--')

plot(df_LPJ_NPP_raw,ax1,'#b2d6ea','-')
plot(df_LPJ_NPP_QM,ax1,'#3292c8',':')
plot(df_LPJ_NPP_QM_CO2,ax1,'#005f95','--')

plot(df_LPJ_tau_raw,ax3,'#b2d6ea','-')
plot(df_LPJ_tau_QM,ax3,'#3292c8',':')
plot(df_LPJ_tau_QM_CO2,ax3,'#005f95','--')

plot(df_CMIP_cVeg,ax5,'#b2d6ea','-')
plot(df_CMIP_NPP,ax4,'#b2d6ea','-')
plot(df_CMIP_tau,ax6,'#b2d6ea','-')

ds_cVeg = xr.open_dataset('../../../Australia_v4.1/CRUJRA_final/cpool_LPJ-GUESS_1901-2018_oz.nc')
ds_NPP = xr.open_dataset('../../../Australia_v4.1/CRUJRA_final/anpp_LPJ-GUESS_1901-2018_oz.nc')
df = pd.DataFrame()
df['cVeg'] = ds_cVeg['VegC'].values.flatten()
df['NPP'] = ds_NPP['Total'].values.flatten()
df = df.rolling(window=10,center=True).mean()
df['Year'] = np.arange(1901,2019)

ax1.plot(df['Year'], df['NPP'],color='#b1b1b1',lw=3, label='LPJ-GUESS (CRUJRA)')
ax2.plot(df['Year'], df['cVeg'],color='#b1b1b1',lw=3, label='LPJ-GUESS (CRUJRA)')

tau = []
time = np.arange(1901,2018)

for i in range(0,116):
    tau_prel = df['cVeg'][i+1]/ (df['NPP'][i+1] - ((df['cVeg'][i+1] - df['cVeg'][i])/ (time[i+1] - time[i])))
    tau.append(tau_prel)

df_tau = pd.DataFrame()
df_tau['tau'] = tau
df_tau = df_tau.rolling(window=10,center=True).mean()
df_tau['Year'] = np.arange(1902,2018)

ax3.plot(df_tau['Year'], df_tau['tau'],color='#b1b1b1',lw=3, label='LPJ-GUESS (CRUJRA)')

ax1.set_ylim(0.5,14)
ax4.set_ylim(0.5,14)
ax2.set_ylim(3,68)
ax5.set_ylim(3,68)
ax3.set_ylim(1.2,15)
ax6.set_ylim(1.2,15)

ax1.set_ylabel('NPP [PgC yr$^{-1}$]')
ax4.set_ylabel('NPP [PgC yr$^{-1}$]')
ax2.set_ylabel('C$_\mathrm{Veg}$ [PgC]')
ax5.set_ylabel('C$_\mathrm{Veg}$ [PgC]')
ax3.set_ylabel('$\\tau$ [yr]')
ax6.set_ylabel('$\\tau$ [yr]')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('d)', loc='left')
ax5.set_title('e)', loc='left')
ax6.set_title('f)', loc='left')

ax1.set_title('Net primary\nproductivity')
ax2.set_title('Carbon stored\nin vegetation')
ax3.set_title('Apparent carbon\nresidence time')

custom_markers = [Line2D([0], [0], linestyle='-', lw=2, color='k'),
                  Line2D([0], [0], linestyle=':', lw=2, color='k'),
                  Line2D([0], [0], linestyle='--', lw=2, color='k'),
                  Line2D([0], [0], linestyle='-', lw=3, color='#b1b1b1'),
                  Patch(facecolor='#b2d6ea', edgecolor='#b2d6ea'),
                  Patch(facecolor='#3292c8', edgecolor='#3292c8'),
                  Patch(facecolor='#005f95', edgecolor='#005f95')
]

ax4.legend(custom_markers, ['Ensemble median (raw)',
                            'Ensemble median (bias-corrected)',
                            'Ensemble median (bias-corrected, CO$_2$ constant)',
                            'LPJ-GUESS (CRUJRA)',
                            'Ensemble spread (raw)',
                            'Ensemble spread (bias-corrected)',
                            'Ensemble spread (bias-corrected, CO$_2$ constant)'
                            ],

           loc='upper center', bbox_to_anchor=(1.8, -0.15),ncol=2, frameon=False)

plt.savefig('cVeg_NPP_tau_SSP585.pdf')
