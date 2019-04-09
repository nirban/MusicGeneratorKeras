from generateData import *
import h5py

X_train, Y_train, X_val, Y_val, X_test, Y_test = generateData()

print("\n Dataset Created ...\n saving file....\n")


with h5py.File("dataset/MaestroDataMat.h5", "a") as f:
    f["X_train"].resize((f["X_train"].shape[0] + X_train.shape[0]), axis = 0)
    f["X_train"][-X_train.shape[0]:] = X_train
    
    f["Y_train"].resize((f["Y_train"].shape[0] + Y_train.shape[0]), axis = 0)
    f["Y_train"][-Y_train.shape[0]:] = Y_train

    f["X_test"].resize((f["X_test"].shape[0] + X_test.shape[0]), axis = 0)
    f["X_test"][-X_test.shape[0]:] = X_test
    
    f["Y_test"].resize((f["Y_test"].shape[0] + Y_test.shape[0]), axis = 0)
    f["Y_test"][-Y_test.shape[0]:] = Y_test

    f["X_val"].resize((f["X_val"].shape[0] + X_val.shape[0]), axis = 0)
    f["X_val"][-X_val.shape[0]:] = X_val
    
    f["Y_val"].resize((f["Y_val"].shape[0] + Y_val.shape[0]), axis = 0)
    f["Y_val"][-Y_val.shape[0]:] = Y_val

f.close()

'''
with h5py.File("dataset/MaestroDataMat.h5", "w") as f:
    f.create_dataset('X_train', data=X_train, maxshape=(None, 50,49), dtype=bool)
    f.create_dataset('Y_train', data=Y_train, maxshape=(None, 50,49), dtype=bool)
    f.create_dataset('X_test', data=X_test, maxshape=(None, 50,49), dtype=bool)
    f.create_dataset('Y_test', data=Y_test, maxshape=(None, 50,49), dtype=bool)
    f.create_dataset('X_val', data=X_val, maxshape=(None, 50,49), dtype=bool)
    f.create_dataset('Y_val', data=Y_val, maxshape=(None, 50,49), dtype=bool)
'''
f.close()


