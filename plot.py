#!/usr/bin/env python

# Module for plotting.

import matplotlib.pylab as plt
# cartopy related libraries
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.io.img_tiles as cimgt
# Local libraries
from apartment_prices import location


def plot_apartments(x, features):
    i = np.where(features == 'latitude')[0][0]
    j = np.where(features == 'longitude')[0][0]
    # Relative deviation in procentage
    sc = plt.scatter(x[j,:], x[i,:], s=0.01, c='grey',
                     label='data',  # cmap=plt.cm.seismic,
                     transform=ccrs.Geodetic())


def plot_contours(figure_handle, x, y, z, colorbarlabel=None):
    CS = plt.contourf(x, y, z,
                      levels=50,
                      cmap=plt.cm.jet, # viridis
                      alpha=0.3,   # vmin=np.max([0, np.min(z)])
                      transform=ccrs.PlateCarree())
    CS2 = plt.contour(CS, levels=CS.levels[::5], # np.arange(60, 105, 5)
                      cmap=plt.cm.jet, # viridis
                      alpha=1,    # vmin=np.max([0, np.min(z)])
                      transform=ccrs.PlateCarree())
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = figure_handle.colorbar(CS)
    if colorbarlabel != None:
        cbar.ax.set_ylabel(r'price/$m^2$ (ksek)')
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)


def plot_map(longitude_lim, latitude_lim, map_quality=11):
    """
    Plot map.
    """
    # Map extent
    eps = 0*0.01
    extent = [longitude_lim[0] - eps, longitude_lim[1] + eps,
              latitude_lim[0] - eps, latitude_lim[1] + eps]
    # Plot map of Stockholm
    map_design = 'StamenTerrain'
    if map_design == 'GoogleTiles':
        request = cimgt.GoogleTiles()
    elif map_design == 'QuadtreeTiles':
        request = cimgt.QuadtreeTiles()
    elif map_design == 'StamenTerrain':
        request = cimgt.StamenTerrain()
    else:
        # Map designs 'OSM' and 'MapboxTiles' do not work
        sys.exit('Untested map design')
    ax = plt.axes(projection=request.crs)
    gl = ax.gridlines(draw_labels=True, alpha=0., linewidth=0, linestyle='-')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'normal'}
    ax.set_extent(extent)
    ax.add_image(request, map_quality, interpolation='spline36')


def plot_sthlm_landmarks():
    """
    Plot landmarks of Stockholm.
    """
    # Ericsson's headquarter
    lat, long = 59.404853, 17.955040
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long, lat - 0.003, 'Ericsson', transform=ccrs.Geodetic())
    # Mall of Scandinavia
    lat, long = 59.370296, 18.004620
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long, lat, 'MoS', transform=ccrs.Geodetic())
    # Sundbyberg
    lat, long = 59.361153, 17.971963
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long - 0.01, lat + 0.001, 'Sundbyberg', transform=ccrs.Geodetic())
    # Sankt Göransgatan
    position = location.get_location_info('Sankt Göransgatan 96')
    plt.plot(position.longitude, position.latitude, 'sk',
             label='J&S', transform=ccrs.Geodetic())
    #plt.text(sankt_goransgatan[-2], sankt_goransgatan[-3] + 0.001, 'J&S', transform=ccrs.Geodetic())
    # Blekingegatan 27
    # Get latitude and longitude for apartment.
    position = location.get_location_info('Blekingegatan 27')
    plt.plot(position.longitude, position.latitude, 'ok',
             label='P', transform=ccrs.Geodetic())
    # Odenplan
    lat, long = 59.342933, 18.049790
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, 'Odenplan', transform=ccrs.Geodetic())
    # Karlaplan
    lat, long = 59.337701, 18.090109
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, 'Karlaplan', transform=ccrs.Geodetic())
    # Jarlaberg
    lat, long = 59.315768, 18.170054
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long - 0.02, lat + 0.001, 'Jarlaberg', transform=ccrs.Geodetic())
    # Globen
    lat, long = 59.293363, 18.083092
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, 'Globen', transform=ccrs.Geodetic())
    # Telefonplan T-bana
    lat, long = 59.298186, 17.996986
    plt.plot(long, lat, '.k', transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, 'Telefonplan', transform=ccrs.Geodetic())
