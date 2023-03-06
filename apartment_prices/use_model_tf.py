"""
# Predict house prices

Provide some basic information about an apartment, and get an estimate of the value of the apartment,
at any desired time!

## Workflow
- Load machine learning (ML) model
- Provide basic apartment information
- Estimate prices in Stockholm!

"""


import argparse
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow

# Local libraries
from apartment_prices import disp, location, plot, time_stuff, video
from apartment_prices.nn_tf import Model_tf


def get_price_on_grid(model, apartment, latitudes, longitudes, features):
    """
    Return apartment prices on a grid, as well as latitudes and longitudes
    """
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    t0 = time.time()
    apartments_on_grid = np.zeros((len(features), len(latitudes) * len(longitudes)), dtype=np.float32)
    k_lat = np.where(features == "latitude")[0][0]
    k_long = np.where(features == "longitude")[0][0]
    k_distance2SthmlCenter = np.where(features == "distance2SthlmCenter")[0][0]
    apartment_index = 0
    for latitude in latitudes:
        for longitude in longitudes:
            apartments_on_grid[:, apartment_index] = apartment.copy()
            apartments_on_grid[k_lat, apartment_index] = latitude
            apartments_on_grid[k_long, apartment_index] = longitude
            apartments_on_grid[k_distance2SthmlCenter, apartment_index] = location.distance_2_sthlm_center(
                latitude, longitude
            )
            apartment_index += 1
    price_grid = model.predict(apartments_on_grid).reshape(len(latitudes), len(longitudes))
    print("Predicting prices on the grid took {:.1f} seconds".format(time.time() - t0))
    return price_grid, longitude_grid, latitude_grid


def get_price_history(model, apartments):
    """
    Return times and corresponding prices for several apartments

    Parameters
    ----------
    model : Model_tf

    apartments : list

    """
    features = model.attributes["features"]
    time_index = np.where(features == "soldDate")[0][0]
    years = range(2013, datetime.now().year + 1 + 1)
    months = range(1, 13)
    times = np.zeros(len(years) * len(months))
    prices = np.zeros((len(years) * len(months), len(apartments)))
    time_counter = 0
    for year in years:
        for month in months:
            time_stamp = time_stuff.get_time_stamp(year, month, 1)
            times[time_counter] = time_stamp
            for j, apartment in enumerate(apartments):
                tmp = apartment.copy()
                tmp[time_index] = time_stamp
                prices[time_counter, j] = model.predict(tmp)
            time_counter += 1
    return times, prices


def plot_price_change_over_time(model, apartments):
    features = model.attributes["features"]
    times, prices = get_price_history(model, apartments.values())
    time_index = np.where(features == "soldDate")[0][0]
    area_index = np.where(features == "livingArea")[0][0]
    fig, axes = plt.subplots(nrows=2, sharex=True)
    for j, (label, apartment) in enumerate(apartments.items()):
        axes[0].plot(times, prices[:, j] / 10**6, "-", label=label)
        axes[1].plot(times, prices[:, j] / (apartment[area_index] * 10**3), "-", label=label)
        # Plot current price
        tmp = apartment.copy()
        time_stamp = datetime.now().timestamp()
        tmp[time_index] = time_stamp
        price_tmp = model.predict(tmp)
        axes[0].plot(time_stamp, price_tmp / 10**6, "o", color="k")
        axes[1].plot(time_stamp, price_tmp / (tmp[area_index] * 10**3), "o", color="k")
    axes[1].set_xlabel("time")
    axes[0].set_ylabel("price (Msek)")
    axes[1].set_ylabel("price/livingArea (ksek/$m^2$)")
    years = range(datetime.fromtimestamp(np.min(times)).year, datetime.fromtimestamp(np.max(times)).year + 1)
    for ax in axes:
        ax.set_xticks([time_stuff.get_time_stamp(year, 1, 1) for year in years])
        ax.set_xticklabels(years)
        ax.grid()
        ax.legend()
    plt.tight_layout()
    plt.savefig("figures/time_evolve_new.pdf")
    plt.savefig("figures/time_evolve_new.png")
    plt.show()


def plot_price_change_with_size(model, apartments):
    feature = "livingArea"
    features = model.attributes["features"]
    area_index = np.where(features == feature)[0][0]
    time_index = np.where(features == "soldDate")[0][0]
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
            price_density[k, j] = model.predict(tmp) / area
    # Plot price density
    plt.figure()
    for j, (label, apartment) in enumerate(apartments.items()):
        plt.plot(areas, price_density[:, j] / 1000, "-", label=label)
        # Plot price density for actual area size, at current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        plt.plot(tmp[area_index], model.predict(tmp) / (tmp[area_index] * 1000), "o", color="k")
    plt.xlabel(feature + "  ($m^2$)")
    plt.ylabel("price/livingArea (ksek/$m^2$)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/price_density_new.pdf")
    plt.savefig("figures/price_density_new.png")
    plt.show()


def plot_price_change_with_floor(model, apartments):
    feature = "floor"
    features = model.attributes["features"]
    floor_index = np.where(features == feature)[0][0]
    area_index = np.where(features == "livingArea")[0][0]
    time_index = np.where(features == "soldDate")[0][0]
    floors = np.arange(11)
    price_density = np.zeros((len(floors), len(apartments)))
    time_stamp = datetime.now().timestamp()
    for j, apartment in enumerate(apartments.values()):
        # Change to current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        for k, floor in enumerate(floors):
            # Change floor
            tmp[floor_index] = floor
            price_density[k, j] = model.predict(tmp) / tmp[area_index]
    # Plot price density vs floor
    plt.figure()
    for j, (label, apartment) in enumerate(apartments.items()):
        plt.plot(floors, price_density[:, j] / 1000, "-", label=label)
        # Plot price density for actual floor, at current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        plt.plot(tmp[floor_index], model.predict(tmp) / (tmp[area_index] * 1000), "o", color="k")
    plt.xlabel(feature)
    plt.ylabel("price/livingArea (ksek/$m^2$)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/price_floor_new.pdf")
    plt.savefig("figures/price_floor_new.png")
    plt.show()


def plot_price_change_with_building_year(model, apartments):
    feature = "constructionYear"
    features = model.attributes["features"]
    construction_index = np.where(features == feature)[0][0]
    area_index = np.where(features == "livingArea")[0][0]
    time_index = np.where(features == "soldDate")[0][0]
    construction_years = np.arange(1900, 2020, 2)
    price_density = np.zeros((len(construction_years), len(apartments)))
    time_stamp = datetime.now().timestamp()
    for j, apartment in enumerate(apartments.values()):
        # Change to current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        for k, construction_year in enumerate(construction_years):
            # Change construction year
            tmp[construction_index] = construction_year
            price_density[k, j] = model.predict(tmp) / tmp[area_index]
    # Plot price density vs floor
    plt.figure()
    for j, (label, apartment) in enumerate(apartments.items()):
        plt.plot(construction_years, price_density[:, j] / 1000, "-", label=label)
        # Plot price density for actual construction year, at current time
        tmp = apartment.copy()
        tmp[time_index] = time_stamp
        plt.plot(tmp[construction_index], model.predict(tmp) / (tmp[area_index] * 1000), "o", color="k")
    plt.xlabel(feature)
    plt.ylabel("price/livingArea (ksek/$m^2$)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/price_construction_new.pdf")
    plt.savefig("figures/price_construction_new.png")
    plt.show()


def plot_price_on_map(model, apartment, latitudes, longitudes, x=None):
    """
    Plot map with contour lines of apartment prices, using current time

    Parameters
    ----------
    model : Model_tf

    apartment : list

    latitudes : ndarray(N)

    longitudes : ndarray(M)

    x : None or ndarray(K,L)
        If not None, represents K features for L different apartments.

    """
    features = model.attributes["features"]
    i = np.where(features == "soldDate")[0][0]
    apartment_reference = apartment.copy()
    # Change to current time
    apartment_reference[i] = datetime.now().timestamp()
    price_grid, longitude_grid, latitude_grid = get_price_on_grid(
        model, apartment_reference, latitudes, longitudes, features
    )
    price_grid[price_grid < 0] = np.nan
    # Plot map and apartment prices
    fig = plt.figure(figsize=(8, 8))
    plot.plot_map([longitudes[0], longitudes[-1]], [latitudes[0], latitudes[-1]], map_quality=12)
    if x is not None:
        plot.plot_apartments(x, features)
    # Plot the price
    i = np.where(features == "livingArea")[0][0]
    plot.plot_contours(
        fig,
        longitude_grid,
        latitude_grid,
        price_grid / (apartment_reference[i] * 10**3),
        colorbarlabel=r"price/$m^2$ (ksek)",
    )
    plot.plot_sthlm_landmarks()
    plt.legend(loc=1)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.savefig("figures/sthlm_new.pdf")
    plt.savefig("figures/sthlm_new.png")
    plt.show()


def videos_on_map(model, apartment, latitudes, longitudes, x=None):
    """
    Videos

    Video of map with contour lines of apartment prices, when vary time.

    Parameters
    ----------
    model : Model_tf

    apartment : list

    latitudes : ndarray(N)

    longitudes : ndarray(M)

    x : None or ndarray(K,L)
        If not None, represents K features for L different apartments.

    """
    years = range(2013, 2021)
    months = range(1, 13)
    days = (1, 15)

    features = model.attributes["features"]
    time_index = np.where(features == "soldDate")[0][0]
    times = np.zeros(len(years) * len(months) * len(days))
    prices = []
    time_counter = 0
    for year in years:
        for month in months:
            for day in days:
                time_stamp = time_stuff.get_time_stamp(year, month, day)
                times[time_counter] = time_stamp
                apartment_reference = apartment.copy()
                # Change time
                apartment_reference[time_index] = time_stamp
                price_grid, longitude_grid, latitude_grid = get_price_on_grid(
                    model, apartment_reference, latitudes, longitudes, features
                )
                price_grid[price_grid < 0] = np.nan
                prices.append(price_grid)
                time_counter += 1
    prices = np.array(prices)
    # Plot the prices
    area_index = np.where(features == "livingArea")[0][0]
    # price per m^2 (ksek)
    prices /= apartment_reference[area_index] * 10**3
    price_change = 100 * ((prices / prices[0]) - 1)

    make_video_on_map(
        times,
        prices,
        longitude_grid,
        latitude_grid,
        features,
        x,
        filename_keyword="sthlm_new",
        colorbarlabel=r"price/$m^2$ (ksek)",
    )
    make_video_on_map(
        times,
        price_change,
        longitude_grid,
        latitude_grid,
        features,
        x,
        filename_keyword="sthlm_change_new",
        colorbarlabel=r"price change  (%)",
    )


def make_video_on_map(
    times, prices, longitude_grid, latitude_grid, features=None, x=None, filename_keyword="sthlm_new", colorbarlabel=""
):
    levels = np.linspace(np.min(prices), np.max(prices), 50)
    longitude_lim = np.min(longitude_grid), np.max(longitude_grid)
    latitude_lim = np.min(latitude_grid), np.max(latitude_grid)
    filenames = []
    for time_stamp, price_grid in zip(times, prices):
        fig = plt.figure(figsize=(8, 8))
        plot.plot_map(longitude_lim, latitude_lim, map_quality=12)
        if x is not None:
            plot.plot_apartments(x, features)
        plot.plot_contours(fig, longitude_grid, latitude_grid, price_grid, levels=levels, colorbarlabel=colorbarlabel)
        plot.plot_sthlm_landmarks()
        plt.legend(loc=1)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        t = datetime.fromtimestamp(time_stamp)
        plt.title("year: {:4d}, month: {:2d}, day: {:2d}".format(t.year, t.month, t.day))
        filename = (
            "figures/video/"
            + filename_keyword
            + "_year"
            + str(t.year)
            + "_month"
            + str(t.month)
            + "_day"
            + str(t.day)
            + ".png"
        )
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
    # Make video from figures
    fps = 32.0
    video.pngs_to_movie(filenames, movie_filename="figures/video/" + filename_keyword + ".mp4", codec="mp4v", fps=fps)
    # video.pngs_to_gif(filenames, movie_filename='figures/video/' + filename_keyword + '.gif', fps=fps)


def plot_distance_to_ceneter_on_map(latitudes, longitudes):
    # Plot figure with distance to Stockholm center
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    d2c_grid = np.zeros_like(longitude_grid, dtype=float)
    for i, lat in enumerate(latitudes):
        for j, long in enumerate(longitudes):
            d2c_grid[i, j] = location.distance_2_sthlm_center(lat, long)
    fig = plt.figure(figsize=(8, 8))
    plot.plot_map([longitudes[0], longitudes[-1]], [latitudes[0], latitudes[-1]], map_quality=12)
    plot.plot_contours(fig, longitude_grid, latitude_grid, d2c_grid, colorbarlabel=r"distance to center  (km)")
    plot.plot_sthlm_landmarks()
    plt.legend(loc=1)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.savefig("figures/sthlm_d2c_new.pdf")
    plt.savefig("figures/sthlm_d2c_new.png")
    plt.show()


def main(ai_name, verbose):
    model = Model_tf(ai_name)
    features = model.attributes["features"]
    y_label = model.attributes["y_label"]
    print("Input features:", features)
    print(f"y_label: {y_label}")

    # Provide basic apartment information
    apartments = {}
    label = "Sankt Göransgatan 96"
    position = location.get_location_info(label)
    print("Location:", position)
    sankt_goransgatan_dict = {
        "soldDate": time_stuff.get_time_stamp(2019, 5, 31),
        "livingArea": 67,
        "rooms": 3,
        "rent": 3370,
        "floor": 4,
        "constructionYear": 1993,
        "latitude": position.latitude,
        "longitude": position.longitude,
        "distance2SthlmCenter": location.distance_2_sthlm_center(position.latitude, position.longitude),
        "ocean": 2564,
    }
    apartments[label] = [sankt_goransgatan_dict[feature] for feature in features]

    label = "Blekingegatan 27"
    position = location.get_location_info(label)
    print("Location:", position)
    blekingegatan_dict = {
        "soldDate": time_stuff.get_time_stamp(2019, 4, 1),
        "livingArea": 44,
        "rooms": 2,
        "rent": 2800,
        "floor": 1.5,
        "constructionYear": 1927,
        "latitude": position.latitude,
        "longitude": position.longitude,
        "distance2SthlmCenter": location.distance_2_sthlm_center(position.latitude, position.longitude),
        "ocean": float("nan"),
    }
    apartments[label] = [blekingegatan_dict[feature] for feature in features]

    # Estimate prices in Stockholm!

    # Print apartment info and predict prices
    for label, apartment in apartments.items():
        print(label)
        disp.apartment_into(features, apartment, model)

    plot_price_change_over_time(model, apartments)

    plot_price_change_with_size(model, apartments)

    plot_price_change_with_floor(model, apartments)

    plot_price_change_with_building_year(model, apartments)

    # Create contour color-map of Stockholm
    # Model the apartment price on a grid of geographical positions.
    # Keep all paramteres fixed except for the position related features
    # (such as latitude and longitude, and distace to Stockholm's center).
    # Examples of possibly interesting parameter values are:
    # - Sankt Göransgatan 96, at the present time
    # - Median apartment in Stockholm, at the present time
    latitude_lim = [59.233, 59.45]
    longitude_lim = [17.82, 18.19]
    latitudes = np.linspace(latitude_lim[0], latitude_lim[1], 301)
    longitudes = np.linspace(longitude_lim[0], longitude_lim[1], 300)
    plot_price_on_map(model, apartments["Sankt Göransgatan 96"], latitudes, longitudes)

    # Movie about how prices on map vary over time
    videos_on_map(model, apartments["Sankt Göransgatan 96"], latitudes, longitudes)

    plot_distance_to_ceneter_on_map(latitudes, longitudes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Neural Network")
    parser.add_argument("ai_name", help="Name of AI model name.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="Verbose flag.")
    args = parser.parse_args()

    # This is deprecated. But gives a 30 times speed-up, when predict price for one apartment.
    # With more apartments the difference is much smaller.
    tensorflow.compat.v1.disable_eager_execution()

    main(args.ai_name, args.verbose)
