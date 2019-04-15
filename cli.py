from network import Network
inputs = [
    [0.3, 0.2, 0.7],
    [0.4, 0.8, 0.3],
    [0.2, 0.5, 0.1],
    [0.9, 0.3, 0.2],
    [0, 0.2, 0.1],
]
outputs = [
    [0, 1, 0],
    [0, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 0],
]

if __name__ == '__main__':
    print('Making network')
    n = Network()
    print('Executing Network')
    n.train(inputs, outputs)
    print('Exiting')
