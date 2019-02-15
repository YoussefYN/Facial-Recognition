import pickle

import cv2
import tensorflow as tf


class Loader:

    def __init__(self, im_size=299):
        self.im_size = im_size
        self._assemble_cnn_graph()
        self._open_session()
        try:
            self.cache = pickle.load(open("img_cach_file.pkl", "rb"))
            print("Re-read Cache")
        except (OSError, IOError):
            self.cache = dict()

    def _assemble_cnn_graph(self):
        pretrained_graph_path = 'InceptionV3.pb'
        with tf.gfile.GFile(pretrained_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            self.input_ = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
            self.cnn_embedding_ = tf.get_default_graph().get_tensor_by_name("Model_Output:0")

    def _load_image(self, path):
        image = cv2.resize(cv2.imread(path), (self.im_size, self.im_size))
        embedding = self.sess.run(self.cnn_embedding_, feed_dict={self.input_: [image]})[0]
        self.cache[path] = embedding
        return embedding

    def load_preprocess_images(self, batch, prefix=""):
        return [self.cache[prefix + p] if prefix + p in self.cache else self._load_image(prefix + p)
                for p in batch]

    def _open_session(self):
        self.sess = tf.Session()

    def _save_cache(self):
        file = open("img_cach_file.pkl", "wb")
        pickle.dump(self.cache, file)
        file.close()
