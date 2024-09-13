import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

ds = xr.open_dataset('vegetation_mask.nc')

fig = plt.figure(figsize=(6,6))
ax=plt.axes(projection=ccrs.PlateCarree())

levels = [1,2,3,4,5,6,7]

cmap = ListedColormap(['#228833', '#CCBB44', '#4477AA',
                       '#66CCEE', '#EE6677', '#AA3377'])

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs=ax.pcolormesh(ds['lon'], ds['lat'], ds.land_cover.values,
                 transform=ccrs.PlateCarree(),
                 cmap=cmap, norm=norm)

ax.coastlines()
ax.axis('off')

ax.add_patch(mpatches.Rectangle(xy=[118, -12], width=10, height=6,
                                facecolor='white',
                                zorder=12,
                                transform=ccrs.PlateCarree())
             )
ax.add_patch(mpatches.Rectangle(xy=[146, -14], width=20, height=10,
                                facecolor='white',
                                zorder=12,
                                transform=ccrs.PlateCarree())
             )

cax = plt.axes([0.1, 0.08, 0.8, 0.03])
cbar=fig.colorbar(cs, orientation='horizontal', cax=cax)
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([1.5,2.5,3.5,4.5,5.5,6.5])
cbar.ax.set_xticklabels(['Tropics', 'Savanna', 'Warm \n temperate',
                         'Cool \n temperate', 'Mediterrenean',
                         'Desert'], ha='center')

#-- call  function and plot figure
plt.subplots_adjust(top=0.97, left=0.00, right=1.00, bottom=0.14, wspace=0.03,
                    hspace=0.08)

# plt.show()
plt.savefig('landcover_classes.png', dpi=300)
