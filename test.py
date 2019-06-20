from net import Pix2Pix
import config
from data_loader import DataLoader
import numpy as np

def test(batch_size = 12):
    model = Pix2Pix
    model.restore(config.model_path)
    dataset_name = 'cifar_images'
    data_loader = DataLoader(dataset_name=dataset_name,
                             img_res = (config.img_rows, config.img_cols))
    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
        fake_A = model.generator.predict(imgs_B)
        imgs_B = imgs_B.repeat(3, -1)
        gen_imgs = np.concat