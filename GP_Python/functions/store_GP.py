import pickle


def store_GP(GP, filename):
    path = 'saved_GPs/'
    file = open(path+filename, 'wb')
    pickle.dump(GP, file)
    file.close()
    pass


def load_GP(filename):
    path = 'saved_GPs/'
    file = open(path+filename, 'rb')
    GP = pickle.load(file)
    return GP


