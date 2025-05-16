import logging
import sys
from dotenv import load_dotenv

load_dotenv()

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
