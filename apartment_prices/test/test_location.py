"""
Module with tests of functions related to location.
"""


import math

# Local
from apartment_prices import location


def test_get_location_info():
    street = "Sankt Göransgatan 96"
    position = location.get_location_info(street)
    assert math.isclose(position.latitude, 59.3341498)
    assert math.isclose(position.longitude, 18.0264807)
