import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

idir='/g/data/w97/lt0205/research/Australia_future/'
ds_mask = xr.open_dataset('../vegetation_mask.nc')
ds_mask = ds_mask.rename({'lat':'Lat', 'lon':'Lon'})

ds_gridarea = xr.open_dataset('../gridarea.nc')

GCMs = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'GFDL-ESM4', 'INM-CM4-8',
        'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'NorESM2-MM']

regions = ['Australia', 'Tropics', 'Savanna', 'Temperate', 'Mediterrenean', 'Desert']

fig = plt.figure(figsize=(10.5,6))

fig.subplots_adjust(hspace=0.22)
fig.subplots_adjust(wspace=0.2)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.12)
fig.subplots_adjust(bottom=0.35)
fig.subplots_adjust(top=0.95)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

def readin(GCM,experiment,file,source,time):
    if source in ('LPJ', 'CMIP6'):
        if source == 'LPJ':
            ds =  xr.open_dataset(idir+'LPJ/SSP'+experiment+'_QM/'+GCM+'/'+file+'_'+
                                  GCM+'_1850-2100.nc')
        elif source == 'CMIP6':
            ds =  xr.open_dataset(idir+'CMIP6/'+file+'_ssp'+experiment+'/'+GCM+'/'+
                                  file+'_day_'+GCM+'_ssp'+experiment+
                                  '_r1i1p1f1_gn_1850-2100_unit.nc')
            ds = ds.rename({'time':'Time', 'lat':'Lat', 'lon':'Lon'})

        if time == 'hist':
            ds = ds.sel(Time=slice('2003','2014')).mean(dim='Time')
        elif time == 'fut':
            ds = ds.sel(Time=slice('2071','2100')).mean(dim='Time')

    elif source == 'Satellite':
        ds =  xr.open_dataset('/g/data/w97/lt0205/research/CAMS_GFAS/CMAS_GFAS_CO2_halfdegree.nc').sel(time=slice('2003','2014'))
        ds = ds.rename({'lat':'Lat', 'lon':'Lon'})

        if time == 'hist':
            ds = ds.groupby('time.year').mean('time')
        elif time == 'fut':
            pass

    return(ds)

def ensemble_average(source,file,time,experiment):
    if source == 'LPJ':
        mean = (readin('ACCESS-CM2',experiment,file,source,time) + \
                readin('ACCESS-ESM1-5',experiment,file,source,time) + \
                readin('CanESM5',experiment,file,source,time) + \
                readin('GFDL-ESM4',experiment,file,source,time) + \
                readin('INM-CM4-8',experiment,file,source,time) + \
                readin('INM-CM5-0',experiment,file,source,time) + \
                readin('IPSL-CM6A-LR',experiment,file,source,time) + \
                readin('KIOST-ESM',experiment,file,source,time) + \
                readin('MIROC6',experiment,file,source,time) + \
                readin('MPI-ESM1-2-HR',experiment,file,source,time) + \
                readin('MPI-ESM1-2-LR',experiment,file,source,time) + \
                readin('MRI-ESM2-0',experiment,file,source,time) + \
                readin('NorESM2-LM',experiment,file,source,time) + \
                readin('NorESM2-MM',experiment,file,source,time)) / 14
    elif source == 'CMIP6':
        mean = (readin('CESM2-WACCM',experiment,file,source,time) + \
                readin('EC-Earth3-Veg',experiment,file,source,time) + \
                readin('MPI-ESM1-2-LR',experiment,file,source,time) + \
                readin('NorESM2-LM',experiment,file,source,time) + \
                readin('NorESM2-MM',experiment,file,source,time)) / 5
    elif source == 'Satellite':
        mean = readin('','','CMAS','Satellite','hist')

    return(mean)

def mask_region(source,file,experiment,region,time):
    ds = ensemble_average(source,file,time,experiment)
    if region == 'Australia':
        ds_masked = ds
    elif region == 'Tropics':
        ds_masked = ds.where(ds_mask['land_cover']==1,0)
    elif region == 'Savanna':
        ds_masked = ds.where(ds_mask['land_cover']==2,0)
    elif region == 'Temperate':
        ds_masked = ds.where((ds_mask['land_cover']==3)|(ds_mask['land_cover']==4),0)
    elif region == 'Mediterrenean':
        ds_masked = ds.where(ds_mask['land_cover']==5,0)
    elif region == 'Desert':
        ds_masked = ds.where(ds_mask['land_cover']==6,0)

    return(ds_masked)

def area_weighted_avg(source,file,var,experiment,region,time):
    ds = mask_region(source,file,experiment,region,time)
    ds = ds * ds_gridarea['cell_area']

    ds_aggr = ds.sum()/1e12

    return(ds_aggr[var].values.tolist())

def stacked_barplot(source,file,var,experiment,time):
    list = []
    for r in regions:
        list.append(area_weighted_avg(source,file,var,experiment,r,time))

    array = np.array(list)
    ratio = array/array[0]
    return(ratio[1:])

total = []
total.append(area_weighted_avg('Satellite','CAMS','co2fire','','Australia','hist'))
total.append(area_weighted_avg('LPJ','cflux','Fire','585','Australia','hist'))
total.append(area_weighted_avg('LPJ','cflux','Fire','585','Australia','fut'))
total.append(area_weighted_avg('CMIP6','fFire','fFire','585','Australia','hist'))
total.append(area_weighted_avg('CMIP6','fFire','fFire','585','Australia','fut'))

ticklabels = ['CAMS-GFAS', 'LPJ_hist', 'LPJ_fut', 'CMIP_hist', 'CMIP_fut']
df_total = pd.DataFrame({'Miau':ticklabels, 'Australia':total})
colors=['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE']
ax1.bar(df_total['Miau'], df_total['Australia'], color=colors)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3,
           frameon=False)
# ax1.set_xticklabels(ticklabels,rotation=90)

df = pd.DataFrame()
df['CAMS-GFAS'] = stacked_barplot('Satellite','CMAS','co2fire','585','hist')
df['LPJ_hist'] = stacked_barplot('LPJ','cflux','Fire','585','hist')
df['LPJ_fut'] = stacked_barplot('LPJ','cflux','Fire','585','fut')
df['CMIP_hist'] = stacked_barplot('CMIP6','fFire','fFire','585','hist')
df['CMIP_fut'] = stacked_barplot('CMIP6','fFire','fFire','585','fut')

df_T = df.T
df_T.columns=regions[1:]
ax2 = df_T.plot(kind='bar', stacked=True, ax=ax2,
                colormap=ListedColormap(colors))

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2,
           frameon=False)

ax1.set_ylabel('Fire CO$_2$ emissions [PgC]')
ax2.set_ylabel('C emission per biome [-]')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

for tick in ax1.get_xticklabels():
    tick.set_rotation(90)

# plt.show()
plt.savefig('fFire_vegetation_SSP585.pdf')
