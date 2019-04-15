from network import Network
from mnist import MNIST

mndata = MNIST('./data')
images, labels = mndata.load_training()
if __name__ == '__main__':
    print('Making network')
    n = Network(
            shape=[(784,), (16,), (16,), (16,), (10,)],
            activations=['relu', 'sigmoid', 'softmax', 'sigmoid'],
            output_labels=list([str(x) for x in range(10)])
    )
    print('Executing Network')
    try:
        n.train(images, labels)
    except KeyboardInterrupt:
        pass
    print('Exiting')
