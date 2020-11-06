

import time
from datetime import datetime


def apartment_into(features, apartment, model=None):
    for feature, value in zip(features, apartment):
        if feature == 'soldDate':
            dt_object = datetime.fromtimestamp(value)
            print(feature + ': {:.2f}'.format(value) + ' (corresponds to date: ' + str(dt_object.date()) + ')')
        else:
            print(feature + ': {:.2f}'.format(value))
    if model != None:
        t0 = time.time()
        print('Predicted apartment price: {:.1f} Msek.'.format(model.predict(apartment) / 10**6))
        print("It took {:.1f} ms to predict the prize.\n".format(1000 * (time.time() - t0)))
    else:
        print('')

