import xarray as xr
import matplotlib.pyplot as plt
import pymannkendall as mk
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(10,10))

ax1=fig.add_subplot(3,3,1)
ax2=fig.add_subplot(3,3,2)
ax3=fig.add_subplot(3,3,3)
ax4=fig.add_subplot(3,3,4)
ax5=fig.add_subplot(3,3,5)
ax6=fig.add_subplot(3,3,6)
ax7=fig.add_subplot(3,3,7)
ax8=fig.add_subplot(3,3,8)
ax9=fig.add_subplot(3,3,9)

def plot_data(axis,var,sim,color,first_year,last_year,location,plot_type):
    if location == 'Australia':
        suffix = 'fldmean'
    else:
        suffix = 'yearmean'
        
    if sim == 'CRUJRA':
        ds_raw = xr.open_dataset('CRUJRA/'+var+'_CRUJRA_'+suffix+'.nc')
        ds_raw = ds_raw.rename({'lat':'Lat','lon':'Lon'})
    elif sim in ('QM','CDFt'):
        ds_raw = xr.open_dataset(sim+'/'+sim+'_'+var+'_NorESM2-MM_'+suffix+'.nc')
    else:
        ds_raw = xr.open_dataset(sim+'/'+sim+'_'+var+'_NorESM2-MM_'+suffix+'.nc')
    
    if plot_type == 'Delta':
        ds_delta = ds_raw - ds_raw.sel(time=slice('1981','2010')).mean(dim='time')
        ds = ds_delta.rolling(time=3, center=True).mean().sel(time=slice('1981','2100'))
    else:
        ds = ds_raw.sel(time=slice(first_year,last_year))
        
    if location == 'Australia':
        pass
    else:
        ds = ds.sel(Lat=-25.75,Lon=116.75,method='nearest')
            
    axis.plot(ds.time.dt.year,ds[var].values.flatten(),color,label=sim)  
    
    secax = axis.twinx()

    # Create a new Axes instance for the scatter plot to the right of the vertical line
    divider = make_axes_locatable(secax)
    axis2 = divider.append_axes("right", size="30%", pad=0.1, sharey=secax)
  
    # Create and fit the model
    if axis in (ax1,ax2,ax3):
        trend = mk.original_test(ds[var].values.flatten())
    else:
        trend = mk.original_test(ds_raw[var].values.flatten())
        
    # Predict trend line
    sign = trend.p
    if sign >= 0.1:
        slope = np.nan
    else:
        slope = trend.slope
    
    if sim == 'CRUJRA':
        x_pos = 0
    elif sim == 'raw':
        x_pos = 0
    elif sim in ('CDFt','QM'):
        x_pos = 0
        
    axis2.scatter(x_pos,slope,color=color)
    
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis2.spines['right'].set_visible(False)
    axis2.spines['top'].set_visible(False)
    secax.spines['right'].set_visible(False)
    secax.spines['top'].set_visible(False)
    secax.set_yticks([])

location=''

sims = ['CRUJRA','raw','CDFt']
axes = [ax1,ax2,ax3]
vars = ['temp','prec','insol']
colors = ['k','#984464','#00678a']

for sim, color in zip(sims,colors):
    for axis,var in zip(axes,vars):
        plot_data(axis,var,sim,color,'1981','2010',location,'')

sims = ['raw','CDFt']
axes = [ax4,ax5,ax6]
vars = ['temp','prec','insol']
colors = ['#984464','#00678a']

for sim, color in zip(sims,colors):
    for axis,var in zip(axes,vars):
        plot_data(axis,var,sim,color,'2071','2100',location,'')

axes = [ax7,ax8,ax9]        
for sim, color in zip(sims,colors):
    for axis,var in zip(axes,vars):
        plot_data(axis,var,sim,color,'','',location,'Delta')

for axis in (ax1,ax2,ax3):
    axis.xaxis.set_ticks([1980,1990,2000,2010,2020])
    axis.set_xticklabels(['1980','1990','2000','2010',''])

for axis in (ax4,ax5,ax6):
    axis.xaxis.set_ticks([2070,2080,2090,2100,2110])
    axis.set_xticklabels(['2070','2080','2090','2100',''])
    
ax1.set_title('Temperature\n\n1981-2010')
ax2.set_title('Precipitation\n\n1981-2010')
ax3.set_title('Incoming SW radiation\n\n1981-2010')

ax4.set_title('2071-2100')
ax5.set_title('2071-2100')
ax6.set_title('2071-2100')

ax7.set_title('2071-2100 vs 1981-2010')
ax8.set_title('2071-2100 vs 1981-2010')
ax9.set_title('2071-2100 vs 1981-2010')

ax1.set_ylabel('T [$^\circ$C]')    
ax4.set_ylabel('T [$^\circ$C]')
ax7.set_ylabel('$\Delta$ T [$^\circ$C]')

ax2.set_ylabel('PPT [mm yr$^{-1}$]')    
ax5.set_ylabel('PPT [mm yr$^{-1}$]')
ax8.set_ylabel('$\Delta$ PPT [mm yr$^{-1}$]')

ax3.set_ylabel('Rad [W m$^{-2}$]')    
ax6.set_ylabel('Rad [W m$^{-2}$]')
ax9.set_ylabel('$\Delta$ Rad [W m$^{-2}$]')

ax1.legend(loc='best',frameon=False)
axes = [ax1,ax2,ax3,
        ax4,ax5,ax6,
        ax7,ax8,ax9]

indices = ['a)','b)','c)',
           'd)','e)','f)',
           'g)','h)','i)']

for axis, index in zip(axes,indices):
    axis.set_title(index,loc='left')

for axis in (ax8,ax9):
    axis.axhline(lw=0.5,color='tab:grey')
    
fig.align_ylabels()
plt.tight_layout()
plt.savefig('oz_hist_monthly_trend.png',dpi=400)