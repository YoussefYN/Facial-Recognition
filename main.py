from siamese_net import SiameseNet
import sys

rand_seed = 91
alpha = 0.2
threshold = -3e-5
minibatch_size = 256
learning_rate = 0.0001
epochs = 10

inception_model = 'InceptionV3.pb'
dataset_path = 'data/dataset'

if len(sys.argv) > 1:
    inception_model = sys.argv[1]

if len(sys.argv) > 2:
    dataset_path = sys.argv[2]

model = SiameseNet(dataset_path, inception_model, learning_rate, seed=rand_seed)
model.train(epochs=epochs, minibatch_size=minibatch_size, alpha=alpha, thresh=threshold)
