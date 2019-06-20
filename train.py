import numpy as np
from visdom import Visdom
import datetime
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from data_loader import DataLoader
import config
from net import Pix2Pix


class Train():

    def __init__(self):
        # ----------------------
        # Show accuracy and loss
        # ----------------------
        self.__vis = Visdom()
        x, y1, y2, y3 = 0, 0, 0, 0
        self.__D_losswin = self.__vis.line(
            X=np.array([x]),
            Y=np.array([y1]),
            opts=dict(title='D loss曲线', showlegend=True)
        )
        self.__G_losswin = self.__vis.line(
            X=np.array([x]),
            Y=np.array([y2]),
            opts=dict(title='G loss曲线', showlegend=True)
        )
        self.__Acc_win = self.__vis.line(
            X=np.array([x]),
            Y=np.array([y3]),
            opts=dict(title='acc准确率曲线', showlegend=True)
        )
        # ----------------------
        # Dataset
        # ----------------------
        self.dataset_name = 'cifar_images'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(config.img_rows, config.img_cols))
        # ----------------------
        # Model
        # ----------------------
        self.model = Pix2Pix()
        # Calculate output shape of D (PatchGAN)
        patch = int(config.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

    def train(self, epochs, batch_size=1, sample_interval=50):
        acc_queue = np.zeros(config.fifo_size, dtype=float)
        acc_idx= 0
        old_loss = [0.0, 0,0]

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                fake_A = self.model.generator.predict(imgs_B)
                avg_acc = np.average(acc_queue)
                if avg_acc < config.acc_threshould:
                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.model.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.model.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    acc_queue[acc_idx] = d_loss[1]
                    acc_idx = (acc_idx + 1) % config.fifo_size
                    old_loss = d_loss
                else:
                    _,real_acc = self.model.discriminator.test_on_batch([imgs_A, imgs_B], valid)
                    _,fake_acc = self.model.discriminator.test_on_batch([fake_A, imgs_B], fake)
                    acc_queue[acc_idx] = 0.5 * (fake_acc + real_acc )
                    old_loss[1] = acc_queue[acc_idx]
                    acc_idx = (acc_idx + 1) % config.fifo_size

                d_loss = old_loss
                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.model.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time

                x = epoch * self.data_loader.n_batches + batch_i
                if x % 10 == 0:  # 每10个batch画一个点
                    # D loss曲线
                    y1 = d_loss[0]
                    self.__vis.line(
                        X=np.array([x]),
                        Y=np.array([y1]),
                        win=self.__D_losswin,
                        update='append'
                    )
                    # G loss曲线
                    y2 = g_loss[0]
                    self.__vis.line(
                        X=np.array([x]),
                        Y=np.array([y2]),
                        win=self.__G_losswin,
                        update='append'
                    )
                    # 准确率曲线
                    y3 = 100 * d_loss[1]
                    self.__vis.line(
                        X=np.array([x]),
                        Y=np.array([y3]),
                        win=self.__Acc_win,
                        update='append'
                    )

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%, avg acc: %3d%%] [G loss: %f] time: %s"
                      % (epoch, epochs,
                         batch_i,
                         self.data_loader.n_batches,
                         d_loss[0],
                         100 * d_loss[1],
                         100 * np.average(acc_queue),
                         g_loss[0],
                         elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    self.model.save(config.model_path)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.model.generator.predict(imgs_B)
        imgs_B = imgs_B.repeat(3, -1)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    t = Train()
    t.train(epochs=20, batch_size=12, sample_interval=200)
