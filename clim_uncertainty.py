import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import signal

fig = plt.figure(figsize=(9,4))

fig.subplots_adjust(hspace=0.22)
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.08)
fig.subplots_adjust(top=0.94)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1=fig.add_subplot(1,3,1)
ax2=fig.add_subplot(1,3,2)
ax3=fig.add_subplot(1,3,3)

GCM_list = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'EC-Earth3',
            'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8',
            'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR',
            'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'NorESM2-MM']

def read_in(GCM,var,scen,BC):
    if var == 'VegC':
        ds = xr.open_dataset('../SSP'+scen+'_'+BC+'/'+GCM+'/cpool_'+GCM+
                             '_1850-2100_oz.nc')
    elif var in ('temp', 'prec'):
        ds = xr.open_dataset('../../CMIP6/'+var+'_'+BC+'_ssp'+scen+'/'+var+'_'+GCM+
                             '_ssp'+scen+'_r1i1p1f1_1850-2100_oz.nc')

    return(ds[var].values.flatten())

def generate_dataframe(var,scen,BC):
    df = pd.DataFrame()

    for GCM in GCM_list:
        df[GCM] = read_in(GCM,var,scen,BC)

    return(df)

df_temp_raw_K = generate_dataframe('temp','245','raw')
df_temp_raw = df_temp_raw_K - 273.15
df_temp_QM_K = generate_dataframe('temp','245','QM')
df_temp_QM = df_temp_QM_K - 273.15

df_prec_raw = generate_dataframe('prec','245','raw')
df_prec_QM_meh = generate_dataframe('prec','245','QM')
df_prec_QM = df_prec_QM_meh/86400

df_VegC_raw = generate_dataframe('VegC','245','raw')
df_VegC_QM = generate_dataframe('VegC','245','QM')

def rolling_avg(df):
    df_anomaly = df-df[:30].mean(axis=0)
    rolling = df_anomaly.rolling(window=30,center=True).mean()
    return(rolling)

def boxplot_plot(df,ax,stat,position,facecolor):
    if stat=='avg':
        df_boxplot = df[-30:].mean(axis=0)

    boxplot=ax.boxplot(df_boxplot,
                       positions=[position],
                       patch_artist=True,
                       widths = .5,
                       medianprops = dict(linestyle='-',
                                          linewidth=2,
                                          color='Yellow'),
                       whiskerprops = dict(linestyle='-',
                                           linewidth=1.5,
                                           color='k'),
                       capprops = dict(linestyle='-',
                                       linewidth=1.5,
                                       color='k'),
                       boxprops = dict(linestyle='-',
                                       linewidth=2,
                                       color='Black',
                                       facecolor=facecolor,
                                       alpha=.7))

boxplot_plot(df_temp_raw,ax1,'avg',0.5,'#088da5')
boxplot_plot(df_temp_QM,ax1,'avg',1.5,'#ed1556')
boxplot_plot(df_prec_raw,ax2,'avg',0.5,'#088da5')
boxplot_plot(df_prec_QM,ax2,'avg',1.5,'#ed1556')
boxplot_plot(df_VegC_raw,ax3,'avg',0.5,'#088da5')
boxplot_plot(df_VegC_QM,ax3,'avg',1.5,'#ed1556')

for a in (ax1,ax2,ax3):
    ax1.set_xticklabels([])

for a in (ax1,ax2,ax3):
    a.axvline(1, linewidth=1,  color='k', alpha=0.5)

titles=['a)', 'b)', 'c)']
axes=[ax1,ax2,ax3]

for a, t in zip(axes, titles):
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.set_title(t, loc = 'left')

ax1.set_ylabel('$\mathrm{T_{\mu,2071-2100}}$ [$^\circ$C]')
ax2.set_ylabel('$\mathrm{PPT_{\mu,2071-2100}}$ [mm]')
ax3.set_ylabel('$\mathrm{C_{Total,\mu,2071-2100}}$ [PgC]')

ax1.set_xticklabels(['Raw', 'QM'])
ax2.set_xticklabels(['Raw', 'QM'])
ax3.set_xticklabels(['Raw', 'QM'])

ax1.set_title('Temp. (2071-2100)')
ax2.set_title('PPT (2071-2100)')
ax3.set_title('C$_\mathrm{Veg}$ (2071-2100)')
fig.align_ylabels()

# plt.show()
plt.savefig('clim_uncertainty_SSP245.pdf')
