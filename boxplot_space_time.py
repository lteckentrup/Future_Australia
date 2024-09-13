import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

fig = plt.figure(figsize=(12,11))

fig.subplots_adjust(hspace=0.2)
fig.subplots_adjust(wspace=0.1)
fig.subplots_adjust(right=0.97)
fig.subplots_adjust(left=0.07)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.93)

ax1=fig.add_subplot(2,3,1)
ax2=fig.add_subplot(2,3,2)

ax4=fig.add_subplot(2,3,4)
ax5=fig.add_subplot(2,3,5)
ax6=fig.add_subplot(2,3,6)

regions = ['Australia', 'Tropics', 'Savanna', 'Temperate', 'Mediterrenean', 'Desert']

def read_in_cor(sim,scen,pred,constraint,sens):
    if sens == 'SE_CO2':
        suffix = '_SE_CO2.csv'
    else:
        suffix = '.csv'

    df_cor = pd.read_csv('csv_Spearman_space/cor_'+sim+'_SSP'+scen+'_'+pred+'_'+constraint+suffix,index_col=0)
    df_pval = pd.read_csv('csv_Spearman_space/pval_'+sim+'_SSP'+scen+'_'+pred+'_'+constraint+suffix,index_col=0)

    df = df_cor.where(df_pval<0.05,np.nan)
    df['Simulation'] = (sim+'_'+constraint+'_'+sens)
    return(df)

def make_boxplot_space(scen,pred,axis):

    palette = ['#00678a','#5eccab','#e6a176']
    df = pd.concat([read_in_cor('Offline',scen,pred,'raw',''),
                    read_in_cor('Offline',scen,pred,'QM',''),
                    read_in_cor('Coupled',scen,pred,'raw','')])
    
    df_long = df.melt(value_vars = ['Australia', 'Tropics', 
                                    'Savanna',  'Temperate',  
                                    'Mediterrenean', 'Desert'],
                                    id_vars=['Simulation'])
    
    medians = df_long.groupby(['variable','Simulation'])['value'].median()
    pd.DataFrame(medians).to_csv(pred+'_'+scen+'_space.csv')
    axis = sns.boxplot(x=df_long['variable'],
                       y=df_long['value'],
                       hue=df_long['Simulation'],
                       flierprops={'marker': 'o',
                                   'markerfacecolor':'white'},
                       palette=palette,
                       ax=axis)
    
    axis.axhline(lw=0.5,c='tab:grey',zorder=0)      

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.set_xticks([0,1,2,3,4,5])
    axis.set_xticklabels([])
    axis.set_xlabel('')
    axis.set_ylabel('')
    axis.legend([],[], frameon=False) 
    axis.set_ylim(-1,1)

ds_mask = xr.open_dataset('../vegetation_mask.nc')

def mask_region(scen,GCM,constraint,sim,region,pred,sens):
    if sens == 'SE_CO2':
        suffix = '_SE_CO2.nc'
    else:
        suffix = '_detrend.nc'

    ds = xr.open_dataset('part_cor/cor_'+sim+'_SSP'+scen+'_'+constraint+'_'+GCM+suffix)
    da = ds[pred]
    if region == 'Australia':
        da_masked = da.where(~ds_mask['land_cover'].isnull())
    elif region == 'Tropics':
        da_masked = da.where(ds_mask['land_cover']==1,np.nan)
    elif region == 'Savanna':
        da_masked = da.where(ds_mask['land_cover']==2,np.nan)
    elif region == 'Temperate':
        da_masked = da.where((ds_mask['land_cover']==3)|(ds_mask['land_cover']==4),np.nan)
    elif region == 'Mediterrenean':
        da_masked = da.where(ds_mask['land_cover']==5,np.nan)
    elif region == 'Desert':
        da_masked = da.where(ds_mask['land_cover']==6,np.nan)

    if pred == 'CO2':
        return(da_masked.values.flatten()*1.4)
    else:
        return(da_masked.values.flatten())

def get_median(scen,sim,constraint,region,pred,sens):
    pred_list = []

    if sim == 'Offline':
        GCMs = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'GFDL-ESM4', 
                'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 
                'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
                'NorESM2-LM', 'NorESM2-MM']
    elif sim == 'Coupled':
        GCMs = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
                'EC-Earth3-Veg', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 
                'MPI-ESM1-2-LR', 'NorESM2-LM', 'NorESM2-MM']   

    for GCM in GCMs:
        r_pred = mask_region(scen,GCM,constraint,sim,region,pred,sens)
        r_pred  = r_pred[~np.isnan(r_pred)]
        pred_list.append(np.median(r_pred))

    df = pd.DataFrame()
    df[pred] = pred_list
    df['Region'] = region
    df['Simulation'] = sim+'_'+constraint+'_'+sens
    return(df)

def make_boxplot_time(axis,pred,scen):
    if pred == 'CO2':
        palette = ['#00678a','#5eccab','#e6a176']
        df = pd.concat([get_median(scen,'Offline','raw','Australia',pred,''),
                        get_median(scen,'Offline','raw','Tropics',pred,''),
                        get_median(scen,'Offline','raw','Savanna',pred,''),
                        get_median(scen,'Offline','raw','Temperate',pred,''),
                        get_median(scen,'Offline','raw','Mediterrenean',pred,''),
                        get_median(scen,'Offline','raw','Desert',pred,''),
                        get_median(scen,'Offline','QM','Australia',pred,''),
                        get_median(scen,'Offline','QM','Tropics',pred,''),
                        get_median(scen,'Offline','QM','Savanna',pred,''),
                        get_median(scen,'Offline','QM','Temperate',pred,''),
                        get_median(scen,'Offline','QM','Mediterrenean',pred,''),
                        get_median(scen,'Offline','QM','Desert',pred,''),
                        get_median(scen,'Coupled','raw','Australia',pred,''),
                        get_median(scen,'Coupled','raw','Tropics',pred,''),
                        get_median(scen,'Coupled','raw','Savanna',pred,''),
                        get_median(scen,'Coupled','raw','Temperate',pred,''),
                        get_median(scen,'Coupled','raw','Mediterrenean',pred,''),
                        get_median(scen,'Coupled','raw','Desert',pred,'')])
    else:
        palette = ['#00678a','#5eccab','#e6a176']
        df = pd.concat([get_median(scen,'Offline','raw','Australia',pred,''),
                        get_median(scen,'Offline','raw','Tropics',pred,''),
                        get_median(scen,'Offline','raw','Savanna',pred,''),
                        get_median(scen,'Offline','raw','Temperate',pred,''),
                        get_median(scen,'Offline','raw','Mediterrenean',pred,''),
                        get_median(scen,'Offline','raw','Desert',pred,''),
                        get_median(scen,'Offline','QM','Australia',pred,''),
                        get_median(scen,'Offline','QM','Tropics',pred,''),
                        get_median(scen,'Offline','QM','Savanna',pred,''),
                        get_median(scen,'Offline','QM','Temperate',pred,''),
                        get_median(scen,'Offline','QM','Mediterrenean',pred,''),
                        get_median(scen,'Offline','QM','Desert',pred,''),
                        get_median(scen,'Coupled','raw','Australia',pred,''),
                        get_median(scen,'Coupled','raw','Tropics',pred,''),
                        get_median(scen,'Coupled','raw','Savanna',pred,''),
                        get_median(scen,'Coupled','raw','Temperate',pred,''),
                        get_median(scen,'Coupled','raw','Mediterrenean',pred,''),
                        get_median(scen,'Coupled','raw','Desert',pred,'')])
    
    df_long = df.melt(value_vars=pred,id_vars=['Region','Simulation'])    

    medians = df_long.groupby(['Region','Simulation'])['value'].median()
    pd.DataFrame(medians).to_csv(pred+'_'+scen+'_time_detrend.csv')

    axis = sns.boxplot(x=df_long['Region'],
                       y=df_long['value'],
                       hue=df_long['Simulation'],
                       palette=palette,
                       flierprops={'marker': 'o',
                                   'markerfacecolor':'white'},
                       ax=axis)
    
    axis.axhline(lw=0.5,c='tab:grey',zorder=0)      

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.set_xticks([0,1,2,3,4,5])
    axis.set_xticklabels(['Australia','Tropics','Savanna','Temperate',
                          'Mediterrenean','Desert'],rotation = 90)
    axis.set_xlabel('')
    axis.set_ylabel('')
    axis.legend([],[], frameon=False) 
    axis.set_ylim(-1,1)

scen='245'
# scen='585'
make_boxplot_space(scen,'prec',ax1)
make_boxplot_space(scen,'temp',ax2)

make_boxplot_time(ax4,'prec',scen)
make_boxplot_time(ax5,'temp',scen)
make_boxplot_time(ax6,'CO2',scen)

custom_markers = [Patch(facecolor='#00678a', edgecolor='#00678a'),
                  Patch(facecolor='#5eccab', edgecolor='#5eccab'),
                  Patch(facecolor='#e6a176', edgecolor='#e6a176')]

ax4.legend(custom_markers, ['LPJ-GUESS\n(raw CMIP6)',
                            'LPJ-GUESS\n(bias corrected CMIP6)',
                            'CMIP6'
                            ],
           loc='upper center', bbox_to_anchor=(1.6, -0.4),ncol=4, frameon=False)

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax4.set_title('c)', loc='left')
ax5.set_title('d)', loc='left')
ax6.set_title('e)', loc='left')

ax1.set_title('Effect of precipitation')
ax2.set_title('Effect of temperature')
ax6.set_title('Effect of CO$_2$')

ax6.set_title('Effect of CO$_2$')

ax1.set_ylabel('Spatial effect')
ax4.set_ylabel('Temporal effect')

ax2.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])

fig.align_ylabels()
# plt.show()
plt.savefig('boxplot_cor_SSP'+scen+'_detrend.pdf')
