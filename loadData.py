import h5py

def loadData():
    f = h5py.File("MaestroData10PercentDyn.h5", 'r')
    X_train = f['X_train']
    Y_train = f['Y_train']
    X_val = f['X_val']
    Y_val = f['Y_val']
    print(f.keys())

    #X_test = f['X_test']
    #Y_test = f['Y_test']
    f.close()
    return X_train, Y_train, X_val, Y_val
