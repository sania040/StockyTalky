# Configuration file
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('COINMARKETCAP_API_KEY')