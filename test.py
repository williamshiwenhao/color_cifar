from net import Pix2Pix
import config
from tqdm import tqdm
from skimage.measure import compare_ssim
from keras.models import load_model
import numpy as np
from glob import glob
import re
import os
import scipy
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

def load_img(path_A, path_B):
    img_A = []
    img_B = []
    resize = (config.img_rows, config.img_cols)
    for i in path_A:
        img = scipy.misc.imread(i, mode='RGB').astype(np.float)
        img = scipy.misc.imresize(img, resize)
        img_A.append(img)
    for i in path_B:
        img = scipy.misc.imread(i, mode='L').astype(np.float)
        img = scipy.misc.imresize(img, resize)
        img = img[:, :, np.newaxis]
        img_B.append(img)
    imgs_A = np.array(img_A) / 127.5 - 1.
    imgs_B = np.array(img_B) / 127.5 - 1.
    return imgs_A, imgs_B



def test(batch_size = 12):
    save_dir = 'datasets/result/'
    os.makedirs(save_dir , exist_ok=True)
    generator = load_model(config.model_path)
    dataset_name = 'cifar_images'
    color_path = glob('./datasets/%s/images_test/*' % (dataset_name))
    bw_path = glob('./datasets/%s/images_test_black/*' % (dataset_name))
    file_name = [re.split(r'[/\\]', str)[-1] for str in bw_path]
    n_batchs = int(config.test_size / batch_size)
    total_num = float(n_batchs * batch_size)
    ave_ssmi = 0.0
    for batch_i in tqdm(range(n_batchs)):
        batch_begin = (batch_i) * batch_size
        batch_end = (batch_i+1) *batch_size
        imgs_A, imgs_B = load_img(color_path[batch_begin:batch_end], bw_path[batch_begin: batch_end])
        fake_A = generator.predict(imgs_B)
        imgs_B = imgs_B.repeat(3, -1)
        # Rescale
        for i in range(batch_size):
            img_A = imgs_A[i]
            img_B = imgs_B[i]
            f_A = fake_A[i]
            ssim = compare_ssim(img_A, f_A, multichannel=True)
            ave_ssmi += ssim / total_num

            gen_imgs = np.array([img_B, f_A, img_A])
            gen_imgs = 0.5 * gen_imgs + 0.5
            titles = ['Gray', 'Generated', 'Original']
            fig, axs = plt.subplots(1, 3)
            for j in range(3):
                axs[j].imshow(gen_imgs[j])
                axs[j].set_title(titles[j])
                axs[j].axis('off')
            fig.savefig(save_dir+file_name[batch_begin+i])
            plt.close()
    print('Average SSIM = %f' % ave_ssmi)

if __name__=='__main__':
    test(20)