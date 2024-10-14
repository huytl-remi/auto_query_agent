# utilities/utils.py
import re

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)
