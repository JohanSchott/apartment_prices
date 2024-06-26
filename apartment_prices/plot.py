"""Module for plotting."""

from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pylab as plt
import numpy as np
from cartopy.io.img_tiles import OSM, GoogleTiles, QuadtreeTiles
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

# Local libraries
from apartment_prices import location, nn, time_stuff


def visualize_data(x_raw, y_raw, features, y_label, normal_apartment_indices):
    # Inspect data
    label = "floor"
    cut_off = 30
    j = np.where(features == label)[0][0]
    mask = x_raw[j, :] > cut_off
    # print(np.shape(x_raw))
    # print(np.median(x_raw[j,:]))
    print("Inspect feature: " + label)
    print("Highest apartments are on these floors:")
    print(x_raw[j, mask])
    print("{:d} apartments are above floor {:d}".format(np.sum(mask), cut_off))
    print("Apartment file indices:")
    print(normal_apartment_indices[mask])
    print()

    # Plot histograms
    plt.figure()
    plt.hist(y_raw, bins=50)
    plt.xlabel(y_label)
    plt.ylabel("apartment distribution")
    # plt.xlim([0,20*10**6])
    plt.show()

    plt.figure()
    plt.hist(np.log(y_raw), bins=50)
    plt.xlabel("log(" + y_label + ")")
    plt.ylabel("apartment distribution")
    # plt.xlim([0,20*10**6])
    plt.show()

    # Histograms of feature data
    for i in range(len(features)):
        print("min and max of", features[i], ":", np.min(x_raw[i, :]), np.max(x_raw[i, :]))
        plt.figure()
        plt.hist(x_raw[i, :], bins=30)
        plt.ylabel("apartment distribution")
        if features[i] == "soldDate":
            tmp = [np.min(x_raw[i, :]), np.max(x_raw[i, :])]
            years = [datetime.fromtimestamp(time_stamp).year for time_stamp in tmp]
            xticks = range(years[0], years[1] + 1)
            xticks_position = [time_stuff.get_time_stamp(year, 1, 1) for year in xticks]
            plt.xticks(xticks_position, xticks)
            plt.xlabel("sold date (year)")
        else:
            plt.xlabel(features[i])
        plt.show()
    print()


def plot_errors(x, y, y_model):
    """
    Plot several figures of the errors between model prediction and data.
    """
    # Relative difference
    rel_diff = (y_model - y) / np.abs(y)
    m = len(y)
    indices_all = nn.shuffle_and_divide_indices(m)
    labels = ["train", "cv", "test"]
    print("\n")
    for label, indices in zip(labels, indices_all):
        print("------", label, "set ------")

        print("Analyze the worst apartment")
        i = np.argmax(np.abs(rel_diff[indices]))
        print("Features:", x[:, indices[i]])
        print("Sold price: ", y[indices[i]], ", model price estimation:", y_model[indices[i]])
        print("Relative difference:", rel_diff[indices[i]], "\n")

        print("Identify the worst apartments")
        mask = np.abs(rel_diff[indices]) > 1.5
        print("Relative differences:", rel_diff[indices[mask]])

        print("Mean relative error (%): ", 100 * np.mean(np.abs(rel_diff[indices])))
        print("Median relative error (%) : ", 100 * np.median(np.abs(rel_diff[indices])))
        print("Standard deviation of relative difference (%): ", 100 * np.std(rel_diff[indices]))
        print("Mean relative difference (%): ", 100 * np.mean(rel_diff[indices]))
        print("Min and max relative difference (%): ", 100 * np.min(rel_diff[indices]), 100 * np.max(rel_diff[indices]))
        print("\n")

    # Distribution of prices
    xlim = (0, 18)
    for label, indices in zip(labels, indices_all):
        plt.figure()
        plt.hist(10**-6 * y[indices], bins=50, range=xlim, label="exact")
        plt.hist(10**-6 * y_model[indices], bins=50, range=xlim, label="model", alpha=0.6)
        plt.xlabel("Apartment price (Msek)")
        plt.ylabel("# apartments")
        plt.legend(loc=0)
        plt.title("Price distribution of " + label + " data")
        plt.show()

    # Distribution of logarithm of prices
    range_min = np.min([np.min(np.log(y)), np.min(np.log(y_model))])
    range_max = np.max([np.max(np.log(y)), np.max(np.log(y_model))])
    for label, indices in zip(labels, indices_all):
        plt.figure()
        plt.hist(np.log(y[indices]), bins=50, range=(range_min, range_max), label="exact")
        plt.hist(np.log(y_model[indices]), bins=50, range=(range_min, range_max), label="model", alpha=0.6)
        plt.xlabel("log(apartment price) (sek)")
        plt.ylabel("apartment distribution")
        plt.legend(loc=0)
        plt.title(label)
        plt.show()

    # Distribution of relative difference
    plt.figure()
    alphas = [1, 0.5, 0.3]
    for label, indices, alpha in zip(labels, indices_all, alphas):
        plt.hist(100 * rel_diff[indices], bins=50, range=(-40, 40), density=True, label=label, alpha=alpha)
    plt.xlabel("Relative deviation (%)")
    plt.ylabel("apartment distribution")
    plt.legend(loc=0)
    plt.show()


def plot_apartments_in_color(x, features, color, colorbarlabel):
    i = np.where(features == "latitude")[0][0]
    j = np.where(features == "longitude")[0][0]
    std = np.std(color)
    # Relative deviation in procentage
    sc = plt.scatter(
        x[j, :],
        x[i, :],
        s=0.01,
        c=color,
        vmin=-2 * std,
        vmax=2 * std,
        cmap=plt.cm.seismic,
        label="data",
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(sc)
    if colorbarlabel is not None:
        cbar.ax.set_ylabel(colorbarlabel)


def plot_apartments(x, features):
    i = np.where(features == "latitude")[0][0]
    j = np.where(features == "longitude")[0][0]
    plt.scatter(x[j, :], x[i, :], s=0.01, c="m", label="data", transform=ccrs.PlateCarree())  # cmap=plt.cm.seismic,


def plot_contours(figure_handle, x, y, z, levels=50, colorbarlabel=None):
    CS = plt.contourf(x, y, z, levels=levels, cmap=plt.cm.jet, alpha=0.05, transform=ccrs.PlateCarree())
    CS2 = plt.contour(CS, levels=CS.levels[::1], cmap=plt.cm.jet, alpha=0.8, transform=ccrs.PlateCarree())
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = figure_handle.colorbar(CS)
    if colorbarlabel is not None:
        cbar.ax.set_ylabel(colorbarlabel)
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)


def plot_map(longitude_lim, latitude_lim, map_quality=11):
    """
    Plot map.

    Parameters
    ----------
    longitude_lim : list(2)

    latitude_lim : list(2)

    map_quality : int
        Map rendering quality.
        6 is very bad, 10 is ok, 12 is good, 13 is very good, 14 excellent.

    """

    # Plot map of Stockholm
    map_design = "GoogleTiles"
    tilers = {
        "GoogleTiles": GoogleTiles(style="street"),  # style="satellite" also looks good
        "OSM": OSM(),
        "QuadtreeTiles": QuadtreeTiles(),
    }
    tiler = tilers[map_design]

    ax = plt.axes(projection=tiler.crs)
    gl = ax.gridlines(draw_labels=True, alpha=0.0, linewidth=0, linestyle="-")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 10, "color": "black"}
    gl.ylabel_style = {"size": 10, "color": "black", "weight": "normal"}

    # Map extent
    eps = 0 * 0.01
    extent = [longitude_lim[0] - eps, longitude_lim[1] + eps, latitude_lim[0] - eps, latitude_lim[1] + eps]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(tiler, map_quality)


def plot_sthlm_landmarks():
    """
    Plot landmarks of Stockholm.
    """
    # Ericsson's headquarter
    lat, long = 59.404853, 17.955040
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Ericsson", transform=ccrs.Geodetic())
    # Mall of Scandinavia
    lat, long = 59.370296, 18.004620
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat, "MoS", transform=ccrs.Geodetic())
    # Sundbyberg
    lat, long = 59.361153, 17.971963
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long - 0.01, lat + 0.001, "Sundbyberg", transform=ccrs.Geodetic())
    # Sankt Göransgatan
    position = location.get_location_info("Sankt Göransgatan 96")
    plt.plot(position.longitude, position.latitude, "sk", label="J&S", transform=ccrs.Geodetic())
    # plt.text(sankt_goransgatan[-2], sankt_goransgatan[-3] + 0.001, 'J&S', transform=ccrs.Geodetic())
    # Blekingegatan 27
    # Get latitude and longitude for apartment.
    position = location.get_location_info("Blekingegatan 27")
    plt.plot(position.longitude, position.latitude, "ok", label="P", transform=ccrs.Geodetic())
    # Odenplan
    lat, long = 59.342933, 18.049790
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Odenplan", transform=ccrs.Geodetic())
    # Karlaplan
    lat, long = 59.337701, 18.090109
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Karlaplan", transform=ccrs.Geodetic())
    # Jarlaberg
    lat, long = 59.315768, 18.170054
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long - 0.02, lat + 0.001, "Jarlaberg", transform=ccrs.Geodetic())
    # Globen
    lat, long = 59.293363, 18.083092
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Globen", transform=ccrs.Geodetic())
    # Telefonplan T-bana
    lat, long = 59.298186, 17.996986
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Telefonplan", transform=ccrs.Geodetic())
    # Huddinge station
    lat, long = 59.236353, 17.978874
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Huddinge", transform=ccrs.Geodetic())
    # Drottningholm
    lat, long = 59.321684, 17.886840
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Drottningholm", transform=ccrs.Geodetic())
    # Täby
    lat, long = 59.444222, 18.074514
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Täby", transform=ccrs.Geodetic())
    # Sollentuna
    lat, long = 59.428485, 17.948356
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Sollentuna", transform=ccrs.Geodetic())
    # Spånga
    lat, long = 59.383285, 17.898908
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Spånga", transform=ccrs.Geodetic())
    # Bromma airport
    lat, long = 59.354626, 17.942596
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat - 0.004, "Airport", transform=ccrs.Geodetic())
    # Millesgården
    lat, long = 59.358678, 18.121680
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long - 0.02, lat + 0.001, "Millesgården", transform=ccrs.Geodetic())
    # Naturhistoriska
    lat, long = 59.368972, 18.053875
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long - 0.02, lat + 0.001, "Naturhistoriska", transform=ccrs.Geodetic())
    # Hökarängen
    lat, long = 59.256956, 18.083102
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Hökarängen", transform=ccrs.Geodetic())
    # Bagarmossen
    lat, long = 59.276264, 18.131451
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long - 0.02, lat + 0.002, "Bagarmossen", transform=ccrs.Geodetic())
    # Skärholmen
    lat, long = 59.277121, 17.907009
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Skärholmen", transform=ccrs.Geodetic())
    # Mörby
    lat, long = 59.398426, 18.036220
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long, lat + 0.001, "Mörby", transform=ccrs.Geodetic())
    # Akalla
    lat, long = 59.415509, 17.913094
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long + 0.001, lat + 0.001, "Akalla", transform=ccrs.Geodetic())
    # Jacobsberg
    lat, long = 59.423529, 17.833041
    plt.plot(long, lat, ".k", transform=ccrs.Geodetic())
    plt.text(long + 0.001, lat + 0.001, "Jacobsberg", transform=ccrs.Geodetic())
