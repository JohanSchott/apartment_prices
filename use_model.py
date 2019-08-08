#!/usr/bin/env python

# # Predict house prices
# Provide some basic information about an apartment, and get an estimate of the value of the apartment, at any desired time!
#
# ## Workflow
# ### 1) Load machine learning (ML) model
# ### 2) Provide basic apartment information
# ### 3) Estimate prices in Stockholm!
# #### 3.1) Analyze specific addresses
# #### 3.2) Create contour color-map of Stockholm


import matplotlib.pylab as plt
import numpy as np
from datetime import datetime
# Local libraries
from apartment_prices import time_stuff
from apartment_prices import location
from apartment_prices.load_nn import load_nn_model_from_file
from apartment_prices import plot


def print_apartment_into(features, apartment, model=None):
    for feature, value in zip(features, apartment):
        if feature == 'soldDate':
            dt_object = datetime.fromtimestamp(value)
            print(feature + ': {:.2f}'.format(value) + ' (corresponds to date: ' + str(dt_object.date()) + ')')
        else:
            print(feature + ': {:.2f}'.format(value))
    if model != None:
        print('Predicted apartment price: {:.1f} Msek \n'.format(model.predict(apartment)/10**6))
    else:
        print('')


def main():
    # 1) Load machine learning (ML) model

    #filename_load = 'models/sthlm_layers9_20_10_10_10_5_5_5_5_5_5_5_1_shuffle0.h5'
    filename_load = 'models/sthlm_layers9_40_30_20_10_10_1_shuffle0.h5'

    model = load_nn_model_from_file(filename_load)
    # List of features expected as input by the model
    features = model.features

    # 2) Provide basic apartment information
    apartments = {}
    label = 'Sankt Göransgatan 96'
    position = location.get_location_info(label)
    print('Location:', position)
    sankt_goransgatan_dict = {'soldDate': time_stuff.get_time_stamp(2019, 5, 31),
                              'livingArea': 67,
                              'rooms': 3,
                              'rent': 3370,
                              'floor': 4,
                              'constructionYear': 1996,
                              'latitude': position.latitude,
                              'longitude': position.longitude,
                              'distance2SthlmCenter': location.distance_2_sthlm_center(position.latitude, position.longitude),
                              'ocean': 2564}
    apartments[label] = [sankt_goransgatan_dict[feature] for feature in features]

    label = 'Blekingegatan 27'
    position = location.get_location_info(label)
    print('Location:', position)
    blekingegatan_dict = {'soldDate': time_stuff.get_time_stamp(2019, 4, 1),
                          'livingArea': 44,
                          'rooms': 2,
                          'rent': 2800,
                          'floor': 1.5,
                          'constructionYear': 1927,
                          'latitude': position.latitude,
                          'longitude': position.longitude,
                          'distance2SthlmCenter': location.distance_2_sthlm_center(position.latitude, position.longitude),
                          'ocean': float('nan')}
    apartments[label] = [blekingegatan_dict[feature] for feature in features]

    # Median apartment in Stockholm
    print('Median apartment in Stockholm')
    label = 'median apartment'
    median_apartment_dict = {'soldDate': time_stuff.get_time_stamp(2016, 11, 1),
                             'livingArea': 58.5,
                             'rooms': 2,
                             'rent': 3091,
                             'floor': 2.0,
                             'constructionYear': 1952,
                             'latitude': 59.33,
                             'longitude': 18.04,
                             'distance2SthlmCenter': location.distance_2_sthlm_center(59.33, 18.04),
                             'ocean': float('nan')}
    apartments[label] = [median_apartment_dict[feature] for feature in features]

    # 3) Estimate prices in Stockholm!
    # 3.1) Analyze specific addresses

    # Print apartment info and predicted prices
    for label, apartment in apartments.items():
        print(label)
        print_apartment_into(features, apartment, model)

    # Time evolve apartments
    i = np.where(features == 'soldDate')[0][0]
    years = range(2013, 2022)
    months = range(1,13)
    times = np.zeros(len(years)*len(months))
    prices = np.zeros((len(years)*len(months), len(apartments)))
    time_counter = 0
    for year in years:
        for month in months:
            time_stamp = time_stuff.get_time_stamp(year, month, 1)
            times[time_counter] = time_stamp
            for j, apartment in enumerate(apartments.values()):
                tmp = apartment.copy()
                tmp[i] = time_stamp
                prices[time_counter,j] = model.predict(tmp)
            time_counter += 1
    # Plot prices
    plt.figure()
    for j, (label, apartment) in enumerate(apartments.items()):
        plt.plot(times, prices[:,j], '-', label=label)
        #plt.plot(times, prices[:,j]*np.max(prices[:,0])/np.max(prices[:,j]), '-', label=label)
        # Plot current price
        tmp = apartment.copy()
        time_stamp = datetime.now().timestamp()
        tmp[i] = time_stamp
        plt.plot(time_stamp, model.predict(tmp), 'o', color='k')
    plt.xlabel('time')
    plt.ylabel('price (sek)')
    plt.xticks([time_stuff.get_time_stamp(year, 1, 1) for year in years], years)
    plt.grid()
    plt.legend()
    plt.savefig('figures/time_evolve_new.pdf')
    plt.savefig('figures/time_evolve_new.png')
    plt.show()

    # 3.2) Create contour color-map of Stockholm
    # Model the apartment price on a grid of geographical positions.
    # Keep all paramteres fixed except for the position related features
    # (such as latitude and longitude, and distace to Stockholm's center).
    # Examples of possibly interesting parameter values are:
    # - Median apartment in Stockholm, at the present/current time

    # Change to current time
    i = np.where(features == 'soldDate')[0][0]
    apartments['median apartment, current time'] = apartments['median apartment'].copy()
    apartments['median apartment, current time'][i] = datetime.now().timestamp()
    # Calculate the price for a latitude and longitude mesh
    latitude_lim = [59.28, 59.42]
    longitude_lim = [17.93, 18.19]
    latitudes = np.linspace(latitude_lim[0], latitude_lim[1], 310)
    longitudes = np.linspace(longitude_lim[0], longitude_lim[1], 300)
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    price_grid = np.zeros_like(longitude_grid, dtype=np.float)
    for i, lat in enumerate(latitudes):
        for j, long in enumerate(longitudes):
            tmp = apartments['median apartment, current time'].copy()
            k = np.where(features == 'latitude')[0][0]
            tmp[k] = lat
            k = np.where(features == 'longitude')[0][0]
            tmp[k] = long
            k = np.where(features == 'distance2SthlmCenter')[0][0]
            tmp[k] = location.distance_2_sthlm_center(lat, long)
            price_grid[i,j] = model.predict(tmp)
    price_grid[price_grid < 0] = np.nan

    # Plot map and apartment prices
    fig = plt.figure(figsize=(10, 10))
    # map rendering quality. 6 is very bad, 10 is ok, 12 is good, 13 is very good, 14 excellent
    map_quality = 13
    plot.plot_map(longitude_lim, latitude_lim, map_quality)
    # Plot the price
    i = np.where(features == 'livingArea')[0][0]
    plot.plot_contours(fig, longitude_grid, latitude_grid,
                       price_grid/(apartments['median apartment, current time'][i]*10**3),
                       colorbarlabel=r'price/$m^2$ (ksek)')
    # Plot landmarks of Stockholm
    plot.plot_sthlm_landmarks()
    # Plot design
    plt.legend()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('figures/sthlm_new.pdf')
    plt.savefig('figures/sthlm_new.png')
    plt.show()


if __name__ == "__main__":
    main()
