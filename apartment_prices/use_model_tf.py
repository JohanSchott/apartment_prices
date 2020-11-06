"""
# Predict house prices
Provide some basic information about an apartment, and get an estimate of the value of the apartment, at any desired time!

## Workflow
### *) Load machine learning (ML) model
### *) Provide basic apartment information
### *) Estimate prices in Stockholm!
#### **) Analyze specific addresses
#### **) Create contour color-map of Stockholm

"""


import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
# Local libraries
from apartment_prices import time_stuff
from apartment_prices import location
from apartment_prices import plot
from apartment_prices import disp
from apartment_prices.nn_tf import Model_tf
from apartment_prices import prepare


def main(ai_name, verbose):
    model = Model_tf(ai_name)
    features = model.attributes['features']
    y_label = model.attributes['y_label']
    print('Input features:', features)

    # Provide basic apartment information
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
                              'distance2SthlmCenter': location.distance_2_sthlm_center(
                                  position.latitude, position.longitude),
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
                          'distance2SthlmCenter': location.distance_2_sthlm_center(
                              position.latitude, position.longitude),
                          'ocean': float('nan')}
    apartments[label] = [blekingegatan_dict[feature] for feature in features]

    # Estimate prices in Stockholm!

    # 4.1) Analyze specific addresses in detail

    # Print apartment info and predicted prices
    for label, apartment in apartments.items():
        print(label)
        disp.apartment_into(features, apartment, model)

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
                import time
                t0 = time.time()
                prices[time_counter,j] = model.predict(tmp)
                print('time:', time.time() - t0)
            time_counter += 1
    # Plot prices
    plt.figure()
    for j, (label, apartment) in enumerate(apartments.items()):
        plt.plot(times, prices[:,j]/10**6, '-', label=label)
        # Plot current price
        tmp = apartment.copy()
        time_stamp = datetime.now().timestamp()
        tmp[i] = time_stamp
        plt.plot(time_stamp, model.predict(tmp)/10**6, 'o', color='k')
    plt.xlabel('time')
    plt.ylabel('price (Msek)')
    plt.xticks([time_stuff.get_time_stamp(year, 1, 1) for year in years], years)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/time_evolve_new.pdf')
    plt.savefig('figures/time_evolve_new.png')
    plt.show()

    # Price/m^2 as function of m^2
    feature = 'livingArea'
    area_index = np.where(features == feature)[0][0]
    time_index = np.where(features == 'soldDate')[0][0]
    areas = np.linspace(20, 150, 300)
    price_density = np.zeros((len(areas), len(apartments)))
    time_stamp = datetime.now().timestamp()
    for j, apartment in enumerate(apartments.values()):
        # Change to current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        for k, area in enumerate(areas):
            # Change area
            tmp[area_index] = area
            price_density[k, j] = model.predict(tmp)/area
    # Plot price density
    plt.figure()
    for j, (label, apartment) in enumerate(apartments.items()):
        plt.plot(areas, price_density[:,j]/1000, '-', label=label)
        # Plot price density for actual area size, at current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        plt.plot(tmp[area_index], model.predict(tmp)/(tmp[area_index]*1000), 'o', color='k')
    plt.xlabel(feature + '  ($m^2$)')
    plt.ylabel('price/livingArea (ksek/$m^2$)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/price_density_new.pdf')
    plt.savefig('figures/price_density_new.png')
    plt.show()

    # 4.2) Create contour color-map of Stockholm
    # Model the apartment price on a grid of geographical positions.
    # Keep all paramteres fixed except for the position related features
    # (such as latitude and longitude, and distace to Stockholm's center).
    # Examples of possibly interesting parameter values are:
    # - Median apartment in Stockholm, at the present/current time

    # Change to current time
    i = np.where(features == 'soldDate')[0][0]
    apartments['Sankt Göransgatan 96, current time'] = apartments['Sankt Göransgatan 96'].copy()
    apartments['Sankt Göransgatan 96, current time'][i] = datetime.now().timestamp()
    # Calculate the price for a latitude and longitude mesh
    latitude_lim = [59.233, 59.45]
    longitude_lim = [17.82, 18.19]
    latitudes = np.linspace(latitude_lim[0], latitude_lim[1], 31)
    longitudes = np.linspace(longitude_lim[0], longitude_lim[1], 30)
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    price_grid = np.zeros_like(longitude_grid, dtype=np.float)
    for i, lat in enumerate(latitudes):
        print("latitude =", lat)
        for j, long in enumerate(longitudes):
            tmp = apartments['Sankt Göransgatan 96, current time'].copy()
            k = np.where(features == 'latitude')[0][0]
            tmp[k] = lat
            k = np.where(features == 'longitude')[0][0]
            tmp[k] = long
            k = np.where(features == 'distance2SthlmCenter')[0][0]
            tmp[k] = location.distance_2_sthlm_center(lat, long)
            price_grid[i,j] = model.predict(tmp)
    price_grid[price_grid < 0] = np.nan
    # Plot map and apartment prices
    fig = plt.figure(figsize=(8,8))
    # map rendering quality. 6 is very bad, 10 is ok, 12 is good, 13 is very good, 14 excellent
    map_quality = 12
    plot.plot_map(longitude_lim, latitude_lim, map_quality)
    # Plot the price
    i = np.where(features == 'livingArea')[0][0]
    plot.plot_contours(fig, longitude_grid, latitude_grid,
                       price_grid / (apartments['Sankt Göransgatan 96, current time'][i] * 10**3),
                       colorbarlabel=r'price/$m^2$ (ksek)')
    # Plot landmarks of Stockholm
    plot.plot_sthlm_landmarks()
    # Plot design
    plt.legend(loc=1)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('figures/sthlm_new.pdf')
    plt.savefig('figures/sthlm_new.png')
    plt.show()

    # Plot figure with distance to Stockholm center
    d2c_grid = np.zeros_like(longitude_grid, dtype=np.float)
    for i, lat in enumerate(latitudes):
        for j, long in enumerate(longitudes):
            d2c_grid[i,j] = location.distance_2_sthlm_center(lat, long)
    fig = plt.figure(figsize=(8,8))
    # map rendering quality. 6 is very bad, 10 is ok, 12 is good, 13 is very good, 14 excellent
    map_quality = 12
    plot.plot_map(longitude_lim, latitude_lim, map_quality)
    # Plot distance
    plot.plot_contours(fig, longitude_grid, latitude_grid,
                       d2c_grid,
                       colorbarlabel=r'distance to center  (km)')
    # Plot landmarks of Stockholm
    plot.plot_sthlm_landmarks()
    # Plot design
    plt.legend(loc=1)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('figures/sthlm_d2c_new.pdf')
    plt.savefig('figures/sthlm_d2c_new.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Neural Network")
    parser.add_argument("ai_name",
                        help="Name of AI model name.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                    help="Verbose flag.")
    args = parser.parse_args()
    main(args.ai_name, args.verbose)

