import time
import sys
import tensorflow as tf
import numpy as np
import os
import glob
from skimage import measure
import cv2

project_path = os.path.abspath("ImageDenoisingGAN-master")

# Add the project path to sys.path
sys.path.append(project_path)

from utils import *
from model import *

# Recursively find all BMP files in dataset from multiple subfolders
bmp_files = []
folders = ["dataset/raw_noisy_image/pictures-v1", 
           "dataset/raw_noisy_image/pictures-v2", 
           "dataset/raw_noisy_image/pictures-v3",
           "dataset/raw_noisy_image/pictures-v4"]  

for folder in folders:
    bmp_files.extend(glob.glob(os.path.join(folder, "**", "*.bmp"), recursive=True))

print(f"✅ Found {len(bmp_files)} BMP images")  # Debugging


def load_bmp_images(image_paths, image_size=(768, 576), batch_size=32):
    """Loads BMP images in batches to prevent MemoryError."""
    print("🔄 Starting image batch loading...")  # Debugging
    for i in range(0, len(image_paths), batch_size):
        batch_files = image_paths[i:i+batch_size]
        batch_images = []

        for img_path in batch_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Warning: Could not read image {img_path}")  # Debugging
                continue  # Skip corrupted files
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0  # Normalize
            batch_images.append(img)

        print(f"✅ Loaded batch {i//batch_size + 1} with {len(batch_images)} images")  # Debugging
        yield np.array(batch_images)  # Return a batch at a time


def train():
    tf.keras.backend.clear_session()  # Reset TensorFlow graph in TF 2.x
    print("🚀 Starting Training...")  # Debugging

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Load images in batches
    image_batches = load_bmp_images(bmp_files, batch_size=BATCH_SIZE)

    # Define input placeholders
    print("📌 Initializing input tensors...")  # Debugging
    gen_in = tf.Variable(tf.random.normal([1, 768, 576, 3]), dtype=tf.float32, name='input_noisy_image')
    real_in = tf.Variable(tf.random.normal([1, 768, 576, 3]), dtype=tf.float32, name='target_noisy_image')

    # Model setup
    print("📌 Building Generator and Discriminator...")  # Debugging
    Gz = generator(gen_in)
    Dx = discriminator(real_in)
    Dg = discriminator(Gz)

    real_in_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), real_in)
    Gz_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), Gz)

    d_loss = -tf.reduce_mean(tf.math.log(Dx) + tf.math.log(1. - Dg))
    g_loss = (ADVERSARIAL_LOSS_FACTOR * -tf.reduce_mean(tf.math.log(Dg)) +
              PIXEL_LOSS_FACTOR * get_pixel_loss(real_in, Gz) +
              STYLE_LOSS_FACTOR * get_style_loss(real_in_bgr, Gz_bgr) +
              SMOOTH_LOSS_FACTOR * get_smooth_loss(Gz))

    # Optimizers
    print("📌 Setting up optimizers...")  # Debugging
    t_vars = tf.compat.v1.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_solver = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_solver = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)

    # Training Session
    print("🔄 Initializing training session...")  # Debugging
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        saver = initialize(sess)
        initial_step = global_step.eval()

        start_time = time.time()
        print("✅ Training started...")  # Debugging

        for index in range(initial_step, N_EPOCHS):
            try:
                input_noisy_images, target_noisy_images = next(image_batches)  # Get next batch
            except StopIteration:
                print("✅ All images processed. Training complete.")  # Debugging
                break

            print(f"🔄 Training Step {index + 1}/{N_EPOCHS} - Processing {input_noisy_images.shape[0]} images")  # Debugging

            _, d_loss_cur = sess.run([d_solver, d_loss], feed_dict={gen_in: input_noisy_images, real_in: target_noisy_images})
            _, g_loss_cur = sess.run([g_solver, g_loss], feed_dict={gen_in: input_noisy_images, real_in: target_noisy_images})

            # Save model and evaluate every SKIP_STEP
            if (index + 1) % SKIP_STEP == 0:
                print(f"📌 Saving model at step {index + 1}...")  # Debugging
                saver.save(sess, CKPT_DIR, index)
                
                print("🔍 Running validation on denoised images...")  # Debugging
                image = sess.run(Gz, feed_dict={gen_in: input_noisy_images})
                image = np.resize(image[7][56:, :, :], [144, 256, 3])

                imsave(f'val_{index+1}', image)

                # Load saved image and calculate metrics
                image = imageio.imread(f"{IMG_DIR}val_{index+1}.png", mode='RGB').astype('float32')
                
                psnr = measure.compare_psnr(input_noisy_images[0], image, data_range=255)
                ssim = measure.compare_ssim(input_noisy_images[0], image, multichannel=True, data_range=255, win_size=11)

                print(f"✅ Step {index + 1}/{N_EPOCHS} | Gen Loss: {g_loss_cur:.4f} | Disc Loss: {d_loss_cur:.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

    print("🏁 Training complete!")  # Debugging


if __name__ == '__main__':
    train()
