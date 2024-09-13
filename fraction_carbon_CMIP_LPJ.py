import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(10,6))

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

GCMs_LPJ = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'EC-Earth3', 'EC-Earth3-Veg',
            'GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
            'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
            'NorESM2-LM', 'NorESM2-MM']

GCMs_CMIP = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5',
             'CESM2-WACCM', 'EC-Earth3-Veg','IPSL-CM6A-LR',
             'MPI-ESM1-2-LR', 'NorESM2-LM', 'NorESM2-MM']

def read_in(var,experiment,GCM,constraint,fert,ensemble):
    if constraint == True:
        if fert == True:
            fname=('../SSP'+experiment+'_QM/'+GCM+'/cpool_'+GCM+
                    '_1850-2100_oz.nc')
        else:
            fname=('../SSP'+experiment+'_QM/'+GCM+'_SE_CO2/cpool_'+GCM+
                    '_1850-2100_oz.nc')
    elif constraint == False:
        if ensemble == 'CMIP':
            fname=('../../CMIP6/'+var+'_ssp'+experiment+'/'+GCM+'/'+var+
                   '_day_'+GCM+'_ssp'+experiment+'_r1i1p1f1_gn_1850-2100_oz.nc')
        else:
            fname=('../SSP'+experiment+'_raw/'+GCM+'/cpool_'+GCM+
                    '_1850-2100_oz.nc')

    ds = xr.open_dataset(fname)
    return(ds[var])

def create_df_cECO(var,experiment,GCM,constraint,fert,ensemble):
    if ensemble == 'CMIP':
        cVeg = 'cVeg'
        cLitter = 'cLitter'
        cSoil = 'cSoil'
    else:
        cVeg = 'VegC'
        cLitter = 'LitterC'
        cSoil = 'SoilC'
    
    ds_cVeg = read_in(cVeg,experiment,GCM,constraint,fert,ensemble)
    ds_cLitter = read_in(cLitter,experiment,GCM,constraint,fert,ensemble)
    ds_cSoil = read_in(cSoil,experiment,GCM,constraint,fert,ensemble)
    
    df = pd.DataFrame()
    if ensemble == 'CMIP':    
        df['time'] = ds_cVeg.time.dt.year
    else:
        df['time'] = ds_cVeg.Time.dt.year
        
    df['cVeg'] = ds_cVeg.values.flatten()
    df['cLitter'] = ds_cLitter.values.flatten()
    df['cSoil'] = ds_cSoil.values.flatten()
    df['cEco'] = df['cVeg'] + df['cLitter'] + df['cSoil']

    df.set_index('time',inplace=True)
    
    df_change = df - df.loc[1981:2010].mean()
    if ensemble == 'CMIP':
        print(df_change)
    df_ratio = df_change.loc[2071:2100].mean()
    return(df_ratio[var]/df_ratio['cEco'])
    
def create_df(experiment,constraint,fert,ensemble):
    cVeg = []
    cLitter = []
    cSoil = []
     
    if ensemble == 'CMIP':   
        for m in GCMs_CMIP:
            cVeg.append(create_df_cECO('cVeg',experiment,m,constraint,fert,ensemble))
            cLitter.append(create_df_cECO('cLitter',experiment,m,constraint,fert,ensemble))
            cSoil.append(create_df_cECO('cSoil',experiment,m,constraint,fert,ensemble))
    else:
        for m in GCMs_LPJ:
            cVeg.append(create_df_cECO('cVeg',experiment,m,constraint,fert,ensemble))
            cLitter.append(create_df_cECO('cLitter',experiment,m,constraint,fert,ensemble))
            cSoil.append(create_df_cECO('cSoil',experiment,m,constraint,fert,ensemble))
            
    df = pd.DataFrame()
    df['C$_\mathrm{Veg}$'] = cVeg
    df['C$_\mathrm{Litter}$'] = cLitter
    df['C$_\mathrm{Soil}$'] = cSoil
    
    return(df)

def make_boxplot(experiment,axis):
    df_raw = create_df(experiment,False,True,'')
    df_raw['Experiment'] = 'LPJ-GUESS\n(raw CMIP6)'
    df_QM = create_df(experiment,True,True,'')
    df_QM['Experiment'] = 'LPJ-GUESS\n(bias corrected\nCMIP6)'
    df_CO2 = create_df(experiment,True,False,'')
    df_CO2['Experiment'] = 'LPJ-GUESS\n(bias corrected\nCMIP6),CO$_2$constant)'
    df_CMIP = create_df(experiment,False,False,'CMIP')
    df_CMIP['Experiment'] = 'CMIP6'
    
    print(df_CMIP)
    df = pd.concat([df_raw,df_QM,df_CO2,df_CMIP])
    df.reset_index(drop=True,inplace=True)

    df_long = pd.melt(df,id_vars=['Experiment'])
    df_long.rename(columns={'variable':'Carbon pool'},inplace=True)

    axis = sns.boxplot(x=df_long['Experiment'],
                       y=df_long['value'],
                       hue=df_long['Carbon pool'],
                       palette = ['#00678a','#5eccab','#e6a176'],
                       ax=axis)
    
    sns.move_legend(axis, 
                    'upper left',
                    ncol=1, 
                    title=None, 
                    frameon=False)

    axis.set_xticks([0,1,2,3])
    axis.set_xticklabels(axis.get_xticklabels(),rotation=90)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.set_xlabel('')
    axis.set_ylabel('')

make_boxplot('245',ax1)
make_boxplot('585',ax2)

ax2.legend([],[], frameon=False)

ax1.set_ylabel('Fraction of change in C$_\mathrm{Eco}$ [-]')

ax1.set_title('a)',loc='left')
ax2.set_title('b)',loc='left')

ax1.set_title('SSP2-4.5')
ax2.set_title('SSP5-8.5')

ax1.set_ylim(-0.05,1.05)
ax2.set_ylim(-0.05,1.05)

plt.tight_layout()
# plt.show()
plt.savefig('cECO_fraction_LPJ_CMIP.png',dpi=400)
