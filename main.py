from siamese_net import SiameseNet

rand_seed = 91
alpha = 0.2
threshold = -3e-5
minibatch_size = 256
learning_rate = 0.0001
epochs = 10

model = SiameseNet('data/dataset', learning_rate, seed=rand_seed)
model.train(epochs=epochs, minibatch_size=minibatch_size, alpha=alpha, thresh=threshold)
