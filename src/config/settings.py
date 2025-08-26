"""Configuration settings for the supply chain risk predictor."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Database
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/supply_chain.db')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Data Collection
REFRESH_INTERVAL_HOURS = int(os.getenv('REFRESH_INTERVAL_HOURS', 24))

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Semiconductor Companies to Track
SEMICONDUCTOR_COMPANIES = {
    'TSMC': '2330.TW',
    'Intel': 'INTC',
    'NVIDIA': 'NVDA',
    'AMD': 'AMD',
    'Qualcomm': 'QCOM',
    'Broadcom': 'AVGO',
    'Texas Instruments': 'TXN',
    'ASML': 'ASML',
    'Applied Materials': 'AMAT',
    'Micron': 'MU'
}