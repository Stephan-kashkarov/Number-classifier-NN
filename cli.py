from network import Network
from mnist import MNIST
import numpy as np

np.seterr(all="raise")

mndata = MNIST('./data')
images, labels = mndata.load_training()
if __name__ == '__main__':
    print('Making network')
    n = Network(
        shape=[(784,), (128,), (10,)],
        activations=['relu', 'sigmoid', 'softmax',],
        output_labels=list([str(x) for x in range(10)]),
        learning_rate=0.001,
    )
    print('Executing Network')
    try:
        n.train(images, labels)
    except KeyboardInterrupt:
        print('bye')
        pass

    print("Testing")
    total = 0
    trues = 0
    images, labels = mndata.load_testing()
    for i, image in enumerate(images):
        if n.execute(image, labels[i], printing=True):
            trues += 1
        total += 1
    print("Testing complete!")
    print(f"{trues} correct out of {total}")

    print('Exiting')
