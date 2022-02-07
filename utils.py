import IPython.display as display
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import json
import os

# Download an image and read it into a NumPy array.
def download(url, max_dim=None, temp_dir="temp/"):
  print(f'Downloading image from {url}')
  name = url.split('/')[-1]
  img_data = requests.get(url).content
  check_if_dir(temp_dir)
  img_path = os.path.join(temp_dir, 'download.jpg')
  with open(img_path, 'wb') as handler:
      handler.write(img_data)
  img = Image.open(img_path)
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  return np.array(img)

def check_if_dir(dir):
  if not os.path.isdir(dir):
    print(f'{dir} does not exist, creating...')
    os.mkdir(dir)

def open_image(path, max_dim=None):
  if not check_if_local(path):
    return download(path, max_dim)
  img = Image.open(path)
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  return np.array(img)

def check_if_local(path):
  if path.startswith("http"):
    return False
  return True

# Display an image
def show(img):
  display.display(Image.fromarray(np.array(img)))

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    f = open(path)
    data = json.load(f)
    return data

def list_web_files(url, ext='png'):
    page = requests.get(url).text
    print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]