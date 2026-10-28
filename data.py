import os
import tensorflow as tf

IMG_SIZE = 256
NUM_CLASSES = 2
BATCH_SIZE = 4

def parse_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')
    mask = tf.cast(mask, tf.float32)
    mask = tf.where(mask > 127.0, 1, 0)
    mask = tf.cast(mask, tf.int32)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.one_hot(mask, NUM_CLASSES)
    return img, mask

def augment(image, mask):

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask

def get_all_files(root_dir):
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):
                all_files.append(os.path.join(root, file))
    return sorted(all_files)

def load_dataset(root, split="train"):
    img_dir  = os.path.join(root, "seg", split, "images")
    mask_dir = os.path.join(root, "seg", split, "labels")
    img_paths  = get_all_files(img_dir)
    mask_paths = get_all_files(mask_dir)
    print(f"[{split}] IMG: {len(img_paths)}  MASK: {len(mask_paths)}")
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if split == "train":                                          
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
