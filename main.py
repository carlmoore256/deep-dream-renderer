import tensorflow as tf
from model import DeepDream, init_model
from run import run_dd_simple
from utils import open_image

if __name__ == "__main__":

    model = init_model(['mixed3', 'mixed9'])

    run_dd_simple(model, )

    # get_tiled_gradients = TiledGradients(dream_model)