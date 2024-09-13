import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def read_in_LPJ(exp,ESM,var_LPJ,var,sens):
    ds = xr.open_dataset(exp+'_QM/'+ESM+sens+'/'+var_LPJ+'_'+ESM+'_1850-2100_oz.nc')
    
    if var == 'treeFrac':
        da_sum = ds['TeNE'] + ds['TeBS']+ ds['IBS'] + ds['TeBE'] + ds['TrBE'] + ds['TrIBE']+ ds['TrBR']
    elif var == 'grassFrac':
        da_sum = ds['C3G'] + ds['C4G']
    
    da = da_sum / ds['Total'] * 100
    return(da)

def get_diff(exp,ESM,var_LPJ,var):
    da_CTRL = read_in_LPJ(exp,ESM,var_LPJ,var,'')
    da_SE_CO2 = read_in_LPJ(exp,ESM,var_LPJ,var,'_SE_CO2')
    
    da_diff = da_CTRL - da_SE_CO2
    return(da_diff.values.flatten())

def get_dataframe(exp,var):
    ESMs = [
        'ACCESS-ESM1-5', 'ACCESS-CM2', 'CanESM5', 'EC-Earth3', 'EC-Earth3-Veg', 
        'GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 
        'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 
        'NorESM2-LM', 'NorESM2-MM'
    ]

    df = pd.DataFrame()

    # Loop over model names and populate DataFrame
    for ESM in ESMs:
        df[ESM] = get_diff(exp, ESM, 'fpc', var)
        
    df['time'] = np.arange(1850,2101,1)
    df.set_index('time',inplace=True)

    arr_min = df.rolling(10,center=True).mean().min(axis=1).values.flatten()
    arr_max = df.rolling(10,center=True).mean().max(axis=1).values.flatten()
    arr_mean = df.rolling(10,center=True).mean().median(axis=1).values.flatten()

    df['min'] = arr_min
    df['max'] = arr_max
    df['mean'] = arr_mean
    
    return(df.loc[2015:2100])

df_treeFrac_SSP245 = get_dataframe('SSP245','treeFrac')
df_grassFrac_SSP245 = get_dataframe('SSP245','grassFrac')
df_treeFrac_SSP585 = get_dataframe('SSP585','treeFrac')
df_grassFrac_SSP585 = get_dataframe('SSP585','grassFrac')

fig=plt.figure(figsize=(8,10))
ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)

ax1.fill_between(np.arange(2015,2101),
                 df_grassFrac_SSP245['min'],
                 df_grassFrac_SSP245['max'],
                 color='#e6a176',
                 alpha=0.5)
ax1.fill_between(np.arange(2015,2101),
                 df_treeFrac_SSP245['min'],
                 df_treeFrac_SSP245['max'],
                 color='#00678a',
                 alpha=0.5)
ax2.fill_between(np.arange(2015,2101),
                 df_grassFrac_SSP585['min'],
                 df_grassFrac_SSP585['max'],
                 color='#e6a176',alpha=0.5)
ax2.fill_between(np.arange(2015,2101),
                 df_treeFrac_SSP585['min'],
                 df_treeFrac_SSP585['max'],
                 color='#00678a',
                 alpha=0.5)

ax1.plot(np.arange(2015,2101),df_grassFrac_SSP245['mean'],color='#e6a176',lw=2)
ax1.plot(np.arange(2015,2101),df_treeFrac_SSP245['mean'],color='#00678a',lw=2)
ax2.plot(np.arange(2015,2101),df_grassFrac_SSP585['mean'],color='#e6a176',lw=2,label='Grass')
ax2.plot(np.arange(2015,2101),df_treeFrac_SSP585['mean'],color='#00678a',lw=2,label='Tree')

ax2.legend(loc='lower left',frameon=False)

for axis in (ax1,ax2,ax3,ax4,ax5,ax6):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

for axis in (ax1,ax2,ax5,ax6):
    axis.axhline(lw=0.5,color='tab:grey')

ax1.set_title('SSP2-4.5\nTrees vs grass')
ax2.set_title('SSP5-8.5\nTrees vs grass')

ax1.set_title('a)',loc='left')
ax2.set_title('b)',loc='left')

ax1.set_ylabel('Percentage of total LAI [%]')

plt.tight_layout()
# plt.show()
plt.savefig('vegetation_composition.png',dpi=400)
