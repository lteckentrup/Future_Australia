import xarray as xr
import joypy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--overlap_factor', type=int, required=True)
args = parser.parse_args()

def create_df(GCM,sim,scen):
    if sim == 'Coupled':
        ds = xr.open_dataset('../../CMIP6/lai_ssp'+scen+'/'+GCM+
                             '/lai_day_'+GCM+'_ssp'+scen+'_r1i1p1f1_gn_1850-2100_annual.nc')
        df = pd.DataFrame()
        df['Historical'] = ds.sel(time=slice('1985','2014')).lai.values.flatten()
        df['Future'] = ds.sel(time=slice('2071','2100')).lai.values.flatten()

    elif sim == 'Offline':
        ds = xr.open_dataset('../SSP'+scen+'_QM/'+GCM+'/alai_'+GCM+'_1850-2100.nc')

        df = pd.DataFrame()
        df['Historical'] = ds.sel(Time=slice('1985','2014')).alai.values.flatten()
        df['Future'] = ds.sel(Time=slice('2071','2100')).alai.values.flatten()

    df[df <= 0] = np.nan

    df['GCM'] = GCM
    return(df)

def create_joyplot(sim,scen,overlap_factor):
    if sim == 'Offline':
        df = pd.concat([create_df('ACCESS-CM2',sim,scen),
                        create_df('ACCESS-ESM1-5',sim,scen),
                        create_df('CanESM5',sim,scen),
                        create_df('EC-Earth3',sim,scen),
                        create_df('EC-Earth3-Veg',sim,scen),
                        create_df('GFDL-CM4',sim,scen),
                        create_df('GFDL-ESM4',sim,scen),
                        create_df('INM-CM4-8',sim,scen),
                        create_df('INM-CM5-0',sim,scen),
                        create_df('IPSL-CM6A-LR',sim,scen),
                        create_df('MIROC6',sim,scen),
                        create_df('MPI-ESM1-2-HR',sim,scen),
                        create_df('MPI-ESM1-2-LR',sim,scen),
                        create_df('MRI-ESM2-0',sim,scen),
                        create_df('NorESM2-LM',sim,scen),
                        create_df('NorESM2-MM',sim,scen)])

    elif sim == 'Coupled':
        df = pd.concat([create_df('ACCESS-ESM1-5',sim,scen),
                        create_df('CanESM5',sim,scen),
                        create_df('EC-Earth3-Veg',sim,scen),
                        create_df('INM-CM4-8',sim,scen),
                        create_df('INM-CM5-0',sim,scen),
                        create_df('IPSL-CM6A-LR',sim,scen),
                        create_df('MPI-ESM1-2-LR',sim,scen),
                        create_df('NorESM2-LM',sim,scen),
                        create_df('NorESM2-MM',sim,scen)])

    xrange = np.arange(-0.2,4.1,0.1)
    fig, axes = joypy.joyplot(df,
                              column=['Historical', 'Future'],
                              by='GCM',
                              ylim='max',
                              overlap=overlap_factor,
                              colormap=[cm.Blues_r,cm.YlOrRd_r],
                              fill=True,
                              linecolor='tab:grey',
                              linewidth=0.5,
                              x_range=xrange,
                              figsize=(4.5,7)
                              )

    axes[-1].axvline(0.231097,lw=1,color='tab:grey',zorder=0)
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('LAI [m$^2$ m$^{-2}$]')
    if sim == 'Offline':
        axes[0].set_title('      a) LPJ-GUESS',loc='left')
    elif sim == 'Coupled':
        axes[0].set_title('         b) CMIP6',loc='left')
    fig.subplots_adjust(bottom=0.08)
    fig.subplots_adjust(top=0.95)

sim=args.sim
scen=args.scen
overlap_factor = args.overlap_factor

create_joyplot(sim,scen,overlap_factor)
plt.savefig('LAI_pdf_'+sim+'_SSP'+scen+'.pdf')