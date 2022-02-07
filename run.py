import os
import json
import time

import tensorflow as tf
from model import DeepDream, TiledGradients, init_model
from utils import deprocess, show, download
from PIL import Image
import random
import glob
import numpy as np



def run_dd_simple(model: DeepDream, 
                    img: Image, steps: int=100,
                    step_size: float=0.01, 
                    step_size_jitter: float=0) -> Image:

  # Convert from uint8 to the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps
    this_step_size = tf.convert_to_tensor(step_size)
    if step_size_jitter > 0:
      this_step_size += tf.random.normal([1], stddev=step_size_jitter)
    loss, img = model(img, run_steps, tf.constant(this_step_size[0]))    
    print (f"Step {step}, loss {loss}")
  result = deprocess(img)
  return result


def run_dd_octaves(model, img: Image, 
                        steps_per_octave: int=100, 
                        step_size: float=0.01, 
                        octaves: int=range(-2,3), 
                        octave_scale: float=1.3) -> Image:
  
  get_tiled_gradients = TiledGradients(model)
  base_shape = tf.shape(img)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  initial_shape = img.shape[:-1]
  img = tf.image.resize(img, initial_shape)
  for octave in octaves:
    # Scale the image based on the octave
    new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
    img = tf.image.resize(img, tf.cast(new_size, tf.int32))
    for _ in range(steps_per_octave):
      gradients = get_tiled_gradients(img)
      img = img + gradients*step_size
      img = tf.clip_by_value(img, -1, 1)

  result = deprocess(img)
  return result


def run_deep_dream_config(config):
    config["time_initiate"] = time.time()
    #   image = download(config["img"], max_dim=config["max_dim"])
    image = Image.open(config["img"])
    model = init_model(config['model_layers'])

    image = run_dd_octaves(
        model, 
        img=image, 
        steps_per_octave=config["steps_per_octave"], 
        step_size=config["step_size"], 
        octaves=range(config["octaves_range"][0], config["octaves_range"][1]),
        octave_scale=config["octaves_scale"])

    image = tf.image.resize(image, tf.shape(image)[:-1])
    image = tf.image.convert_image_dtype(image/255.0, dtype=tf.uint8)

    if config["save_dir"] is not None:
        file_num = 0
        filename = f'{file_num}-{config["symbol"]}-{config["collection"]}'
        img_path = os.path.join(config["save_dir"], f'{filename}.jpg')
        meta_path = os.path.join(config["save_dir"], f'{filename}.json')

        while os.path.isfile(img_path) or os.path.isfile(meta_path):
            file_num += 1
            filename = f'{file_num}-{config["symbol"]}-{config["collection"]}'
            img_path = os.path.join(config["save_dir"], f'{filename}.jpg')
            meta_path = os.path.join(config["save_dir"], f'{filename}.json')

        tf.keras.utils.save_img(img_path, image)
        config["time_complete"] = time.time()
        config["render_time"] = config["time_complete"] - config["time_initiate"]
        with open(meta_path, 'w') as outfile:
            json.dump(config, outfile, indent=2)

    return image, config

def run_single_image(config):
    config = {
        "img": '/home/carl/back-of-the-router/out/23.png',
        "name": "bionic",
        "collection": "back-of-the-router",
        "symbol": "BOTR",
        "website": "homonculi.org/art",
        "max_dim": 500,
        "steps_per_octave": 10,
        "step_size":0.01,
        "octaves_range":(-3,3),
        "octaves_scale":1.3,
        "model_layers": [16],
        "save_dir": "out/"
    }
    image, config = run_deep_dream_config(config)


def run_batch_configs(batch_config, shots=100):
    if batch_config["save_dir"] is not None:
        if not os.path.exists(batch_config["save_dir"]):
            os.mkdir(batch_config["save_dir"])
    
    img_files = glob.glob(f'{batch_config["img_dir"]}/*.png')
    print(f'{len(img_files)} image files found')
    
    for i, im in enumerate(img_files):
        for s in range(shots):
            shot_config = batch_config.copy()
            shot_config['img'] = im
            shot_config["steps_per_octave"] = random.choice(batch_config["steps_per_octave"])
            shot_config["step_size"] = random.choice(batch_config["step_size"])
            shot_config["octaves_range"] = random.choice(batch_config["octaves_range"])
            shot_config["octaves_scale"] = random.choice(batch_config["octaves_scale"])
            shot_config["model_layers"] = [random.choice(batch_config["model_layers"])]
            print(f'\n=> Running DeepDream config #{s+1}/{shots} file {i}/{len(img_files)}')
            print(f'\n=> {shot_config}')
            run_deep_dream_config(shot_config)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", )
    batch_config = {
        "img_dir" : '/home/carl/back-of-the-router/out/',
        "name": "",
        "collection": "Back-of-the-Router",
        "symbol": "BOTR",
        "website": "homonculi.org/art",
        "max_dim": 500,
        "steps_per_octave": list(range(1, 30)),
        "step_size": np.linspace(0.1, 1e-5, num=100),
        "octaves_range":[(-2,2),(-1,2),(0,2),(1,3)],
        "octaves_scale":[1.1, 1.2, 1.3],
        "model_layers":  [3,6,7,8,9,10,11,12,13,14,15,16],
        "save_dir" : 'out/',
        # "save_dir": "/content/drive/MyDrive/deep-dream-homunculi/SZT-flaimes/",
        "batch_run" : str(time.time()),
        "batch_tag" : "first run of suzani-tapestries with homunculi-deep-dream interface!"
    }

    run_batch_configs(batch_config, shots=10)
