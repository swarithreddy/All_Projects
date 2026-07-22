import os, sys
from dotenv import load_dotenv

def path(file):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, file)
    return file

load_dotenv(path(".env"))