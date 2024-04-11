#  Copyright (c) 2024. Diego Urrutia-Astorga <durrutia@ucn.cl>

import logging as log

import coloredlogs
from dotenv import load_dotenv

# Configure logging
coloredlogs.install(level='DEBUG',
                    field_styles={
                        'asctime': {'color': 'white'},
                        'levelname': {'color': 'white', 'bold': True},
                        'name': {'color': 'blue'}})

log.debug("Initalizing the Tammy application ..")

# Load environment variables
load_dotenv()

log.debug("Initialization complete.")
