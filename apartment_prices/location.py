import numpy as np
from math import pi
from geopy.geocoders import Nominatim


def get_location_info(street, city="Stockholm", county="Stockholms län", country="Sweden"):
    geolocator = Nominatim(user_agent="apartment_prices")  # timeout=5
    location = geolocator.geocode(query={"street": street, "city": city, "county": county, "country": country})
    return location


def distance_2_sthlm_center(latitude, longitude):
    """
    Return distance from position given by latitude and longitude to Stockholm city center (in km).
    """
    # Stockholm city center coordinates.
    latitude_c = 59.33077
    longitude_c = 18.059101
    # Calculate distance to the center
    distance = get_great_circle_distance(latitude, longitude, latitude_c, longitude_c)
    return distance


def get_great_circle_distance(latitude, longitude, latitude_c, longitude_c):
    """
    Return the great circle distance to point c.

    Parameters
    ----------
    latitude : float or array(N)
    longitude : float or array(N)
    latitude_c : float
    longitude_c : float

    """
    # Convert to spherical coordinates theta and phi
    theta, phi = get_spherical_coordinates(latitude, longitude)
    theta_c, phi_c = get_spherical_coordinates(latitude_c, longitude_c)
    distance = get_great_circle_distance_using_spherical_coordinates(theta, phi, theta_c, phi_c)
    return distance


def get_great_circle_distance_using_spherical_coordinates(theta, phi, theta_c, phi_c, r=6371.0):
    """
    Return the great circle distance to point c.

    Parameters
    ----------
    theta :  float or array(N)
    phi : float or array(N)
    theta_c : float
    phi_c : float
    r : float or array(N)
        Default value is the radius of the earth in km.

    """
    # Angle between point and point c.
    dalpha = np.arccos(np.sin(theta) * np.sin(theta_c) * np.cos(phi - phi_c) + np.cos(theta) * np.cos(theta_c))
    return r * dalpha


def get_cartesian_distance(latitude, longitude, latitude_c, longitude_c):
    """
    Return the cartesian distance to point c.

    Parameters
    ----------
    latitude :  float or array(N)
    longitude : float or array(N)
    latitude_c : float
    longitude_c : float

    """
    pos = get_cartesian_coordinates(latitude, longitude)
    pos_c = get_cartesian_coordinates(latitude_c, longitude_c)
    # Calculate the distance to the center point
    distance = np.zeros_like(latitude, dtype=np.float)
    if distance.ndim == 0:
        for i in range(3):
            distance += (float(pos[i, :]) - float(pos_c[i])) ** 2
        distance = float(np.sqrt(distance))
    else:
        for i in range(3):
            distance += (pos[i, :] - float(pos_c[i])) ** 2
        distance = np.sqrt(distance)
    return distance


def get_spherical_coordinates(latitude, longitude):
    """
    Return spherical coordinates.

    Parameters
    ----------
    latitude : float or array(N)
    longitude : float or array(N)

    """
    # Convert to spherical coordinates theta and phi
    theta = pi / 2 - latitude * pi / 180
    phi = np.zeros_like(longitude, dtype=np.float)
    if phi.ndim == 0:
        if longitude >= 0:
            phi = longitude * pi / 180
        else:
            phi = 2 * pi + longitude * pi / 180
    else:
        mask = longitude >= 0
        phi[mask] = longitude[mask] * pi / 180
        phi[np.logical_not(mask)] = 2 * pi + longitude[np.logical_not(mask)] * pi / 180
    return theta, phi


def get_cartesian_coordinates(latitude, longitude):
    # Convert to spherical coordinates theta and phi
    theta, phi = get_spherical_coordinates(latitude, longitude)
    pos = spherical2cartesian_coordinates(theta, phi)
    return pos


def spherical2cartesian_coordinates(theta, phi, r=6371.0):
    """
    Return cartesian coordinates.

    Parameters
    ----------
    theta : float or array(N)
    phi : float or array(N)
    r : float or array(N)
        Default value is the radius of the earth in km.

    """
    pos_x = r * np.sin(theta) * np.cos(phi)
    pos_y = r * np.sin(theta) * np.sin(phi)
    pos_z = r * np.cos(theta)
    pos = np.atleast_2d([pos_x, pos_y, pos_z]).T
    return pos
