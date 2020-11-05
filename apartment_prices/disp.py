
from datetime import datetime

def apartment_into(features, apartment, model=None):
    for feature, value in zip(features, apartment):
        if feature == 'soldDate':
            dt_object = datetime.fromtimestamp(value)
            print(feature + ': {:.2f}'.format(value) + ' (corresponds to date: ' + str(dt_object.date()) + ')')
        else:
            print(feature + ': {:.2f}'.format(value))
    if model != None:
        print('Predicted apartment price: {:.1f} Msek \n'.format(model.predict(apartment) / 10**6))
    else:
        print('')

