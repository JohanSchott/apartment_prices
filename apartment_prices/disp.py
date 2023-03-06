import time
from datetime import datetime

import numpy as np


def apartment_into(features, apartment, model=None):
    for feature, value in zip(features, apartment):
        if feature == "soldDate":
            dt_object = datetime.fromtimestamp(value)
            print(feature + ": {:.2f}".format(value) + " (corresponds to date: " + str(dt_object.date()) + ")")
        else:
            print(feature + ": {:.2f}".format(value))
    if model is not None:
        area_index = np.where(features == "livingArea")[0][0]
        area = apartment[area_index]
        t0 = time.time()
        price = model.predict(apartment)
        print("It took {:.1f} ms to predict the prize.".format(1000 * (time.time() - t0)))
        print(
            "Predicted apartment price at sold date is {:.1f} Msek. Corresponds to {:.1f} ksek/m2.".format(
                price / 10**6, price / (area * 10**3)
            )
        )
        time_index = np.where(features == "soldDate")[0][0]
        apartment_current_time = apartment.copy()
        apartment_current_time[time_index] = datetime.now().timestamp()
        price = model.predict(apartment_current_time)
        print(
            "Predicted apartment price now is {:.1f} Msek. Corresponds to {:.1f} ksek/m2.\n".format(
                price / 10**6, price / (area * 10**3)
            )
        )

    else:
        print("")
