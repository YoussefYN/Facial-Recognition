import os
import pickle
from random import randint

import cv2
import tensorflow as tf
from pandas import read_csv


class Loader:

    def __init__(self, data_path, path, im_size=299):
        self._model_path = path
        self._img_size = im_size
        self._assemble_cnn_graph()
        self._open_session()
        self._data_path = data_path
        self._images_dirs = [[os.path.join(x, y) for y in os.listdir(os.path.join(self._data_path, x))]
                             for x in os.listdir(self._data_path)]
        self._img_idxs = [0] * len(self._images_dirs)
        try:
            self._cache = pickle.load(open("img_cach_file.pkl", "rb"))
            print("Re-read Cache")
        except (OSError, IOError):
            self._cache = dict()

    def _assemble_cnn_graph(self):
        with tf.gfile.GFile(self._model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            self._input = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
            self._cnn_embedding = tf.get_default_graph().get_tensor_by_name("Model_Output:0")

    def _load_image(self, path):
        image = cv2.resize(cv2.imread(os.path.join(self._data_path, path)), (self._img_size, self._img_size))
        embedding = self._sess.run(self._cnn_embedding, feed_dict={self._input: [image]})[0]
        self._cache[path] = embedding
        return embedding

    def _load_preprocess_images(self, batch):
        return [self._cache[p] if p in self._cache else self._load_image(p)
                for p in batch]

    def _open_session(self):
        self._sess = tf.Session()

    def _save_cache(self):
        file = open("img_cach_file.pkl", "wb")
        pickle.dump(self._cache, file)
        file.close()

    def get_next_batch(self, minibatch_sz):
        dirs_len = len(self._images_dirs)
        anc_batch = []
        pos_batch = []
        neg_batch = []
        for i in range(dirs_len):
            images = self._images_dirs[i]
            length = len(images)
            if length < 2:
                continue
            anc_batch.append(images[self._img_idxs[i]])
            pos_batch.append(images[(self._img_idxs[i] + 1) % length])
            self._img_idxs[i] = (self._img_idxs[i] + 1) % length
            rand_class = i
            while (len(self._images_dirs[rand_class]) == 0) or (rand_class == i):
                rand_class = randint(0, dirs_len - 1)
            neg_batch.append(self._images_dirs[rand_class][self._img_idxs[rand_class]])

        def divide(x):
            len_x = len(x)
            return [x[j: min(j + minibatch_sz, len_x)]
                    for j in range(0, len_x, minibatch_sz)]

        return divide(self._load_preprocess_images(anc_batch)), \
               divide(self._load_preprocess_images(pos_batch)), \
               divide(self._load_preprocess_images(neg_batch))

    def get_test_batches(self):
        test_data = read_csv('test_set.csv')
        test_anc_batch = self._load_preprocess_images(
            test_data['Anchor'].values)
        test_pos_batch = self._load_preprocess_images(
            test_data['Positive'].values)
        test_neg_batch = self._load_preprocess_images(
            test_data['Negative'].values)
        return test_anc_batch, test_pos_batch, test_neg_batch
