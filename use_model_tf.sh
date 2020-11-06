#!/bin/bash

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Activate virtual environment and append to PYTHONPATH.
source env.sh

# ----Kungsholmen-----
# Model 1
#python -m apartment_prices.use_model_tf kungsholmen1

# ------Stor-Stockholm--------
# Model 1
#python -m apartment_prices.use_model_tf stor_stockholm1

# Model 2
python -m apartment_prices.use_model_tf stor_stockholm2



