from network import Network

if __name__ == '__main__':
    print('Making network')
    n = Network()
    print('Executing Network')
    n.execute([0.1, 0.2, 0.7], [1,0,0])
    print('Exiting')
