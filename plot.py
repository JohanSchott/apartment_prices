#!/usr/bin/env python

# Module for plotting.

import matplotlib.pylab as plt
import numpy as np
from datetime import datetime
# cartopy related libraries
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.io.img_tiles as cimgt
# Local libraries
from . import location
from . import time_stuff

def visualize_data(x_raw, y_raw, features, y_label, normal_apartment_indices):
    # Inspect data
    label = 'floor'
    cut_off = 30
    j = np.where(features == label)[0][0]
    mask = x_raw[j,:] > cut_off
    #print(np.shape(x_raw))
    #print(np.median(x_raw[j,:]))
    print('Inspect feature: ' + label)
    print('Highest apartments are on these floors:')
    print(x_raw[j,mask])
    print('{:d} apartments are above floor {:d}'.format(np.sum(mask), cut_off))
    print('Apartment file indices:')
    print(normal_apartment_indices[mask])
    print()

    # Plot histograms
    plt.figure()
    plt.hist(y_raw, bins=50)
    plt.xlabel(y_label)
    plt.ylabel('apartment distribution')
    #plt.xlim([0,20*10**6])
    plt.show()

    plt.figure()
    plt.hist(np.log(y_raw), bins=50)
    plt.xlabel('log(' + y_label + ')')
    plt.ylabel('apartment distribution')
    #plt.xlim([0,20*10**6])
    plt.show()

    # Histograms of feature data
    for i in range(len(features)):
        print('min and max of', features[i], ':', np.min(x_raw[i,:]), np.max(x_raw[i,:]))
        plt.figure()
        plt.hist(x_raw[i,:], bins=30)
        plt.ylabel('apartment distribution')
        if features[i] == 'soldDate':
            tmp = [np.min(x_raw[i,:]), np.max(x_raw[i,:])]
            years = [datetime.fromtimestamp(time_stamp).year for time_stamp in tmp]
            xticks = range(years[0], years[1]+1)
            xticks_position = [time_stuff.get_time_stamp(year, 1, 1) for year in xticks]
            plt.xticks(xticks_position, xticks)
            plt.xlabel('sold date (year)')
        else:
            plt.xlabel(features[i])
        plt.show()
    print()


def plot_errors(x_raw, y_raw, y_model, normal_apartment_indices):
    """
    Plot several figures of the errors between model prediction and data.
    """
    # Relative deviation in procentage
    rel_deviation = (y_model - y_raw)/np.abs(y_raw)
    print('Mean relative error on the full data set (%): ', 100*np.mean(np.abs(rel_deviation)))
    print('Median relative error on the full data set (%) : ', 100*np.median(np.abs(rel_deviation)))
    print('Standard deviation of relative deviation (%): ', 100*np.std(rel_deviation))
    print('Mean relative devation (%): ', 100*np.mean(rel_deviation))
    print('Min and max relative deviation (%): ', 100*np.min(rel_deviation), 100*np.max(rel_deviation))

    print('\n Analyze the worst apartment')
    i = np.argmax(np.abs(rel_deviation))
    print('csv index:', normal_apartment_indices[i])
    print('x_raw index:', i)
    print('features:', x_raw[:,i])
    print('sold price: ', y_raw[i], ', model price estimation:', y_model[i])
    print('rel deviation:', rel_deviation[i])

    print('\n Analyze the worst apartments')
    mask = np.abs(rel_deviation) > 1.5
    print(rel_deviation[mask])
    print('csv indices:', normal_apartment_indices[mask])
    print('x_raw indices:', np.where(mask))

    plt.figure()
    plt.plot(y_raw, 'o',label='exact')
    plt.plot(y_model, 'o', label='model')
    plt.xlabel('apartment index')
    plt.ylabel('apartment price (sek)')
    plt.legend(loc=0)
    plt.show()

    plt.figure()
    plt.hist(y_raw, bins=50, range=(0,18*10**6), label='exact')
    plt.hist(y_model, bins=50, range=(0,18*10**6), label='model', alpha=0.6)
    plt.xlabel('apartment price (sek)')
    plt.ylabel('apartment distribution')
    plt.legend(loc=0)
    plt.show()

    plt.figure()
    range_min = np.min([np.min(np.log(y_raw)),np.min(np.log(y_model))])
    range_max = np.max([np.max(np.log(y_raw)),np.max(np.log(y_model))])
    plt.hist(np.log(y_raw), bins=50, range=(range_min, range_max), label='exact')
    plt.hist(np.log(y_model), bins=50, range=(range_min, range_max), label='model', alpha=0.6)
    plt.xlabel('log(apartment price) (sek)')
    plt.ylabel('apartment distribution')
    plt.legend(loc=0)
    plt.show()

    plt.figure()
    plt.plot(100*rel_deviation, 'o')
    plt.xlabel('apartment index')
    plt.ylabel('Relative deviation (%)')
    plt.show()

    plt.figure()
    plt.hist(100*rel_deviation, bins=50, range=(-40, 40))
    plt.xlabel('Relative deviation (%)')
    plt.ylabel('apartment distribution')
    plt.show()

    plt.figure()
    plt.plot(100*np.abs(rel_deviation), 'o')
    plt.xlabel('apartment index')
    plt.ylabel('Relative error (%)')
    plt.show()

    plt.figure()
    plt.hist(100*np.abs(rel_deviation), bins=50, range=(0, 40))
    plt.xlabel('Relative error (%)')
    plt.ylabel('apartment distribution')
    plt.show()


def plot_apartments_in_color(x, features, color, colorbarlabel):
    i = np.where(features == 'latitude')[0][0]
    j = np.where(features == 'longitude')[0][0]
    # Relative deviation in procentage
    sc = plt.scatter(x[j,:], x[i,:], s=0.01, c=color,
                     vmin=-2*np.std(color), vmax=2*np.std(color),
                     cmap=plt.cm.seismic, label='apartments',
                     transform=ccrs.Geodetic())
    cbar = plt.colorbar(sc)
    if colorbarlabel != None:
        cbar.ax.set_ylabel(colorbarlabel)


def plot_apartments(x, features):
    i = np.where(features == 'latitude')[0][0]
    j = np.where(features == 'longitude')[0][0]
    # Relative deviation in procentage
    sc = plt.scatter(x[j,:], x[i,:], s=0.01, c='m',
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
        cbar.ax.set_ylabel(colorbarlabel)
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
