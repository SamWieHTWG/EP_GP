import pickle


def store_GP(GP, filename):
    path = '/home/samuel/Documents/EP_GP/GP_Python/saved_GPs/'
    file = open(path+filename, 'wb')
    pickle.dump(GP, file)
    file.close()
    pass


def load_GP(filename):
    path = '/home/samuel/Documents/EP_GP/GP_Python/saved_GPs/'
    file = open(path+filename, 'rb')
    GP = pickle.load(file)
    #print(GP)
    file.close()
    return GP


