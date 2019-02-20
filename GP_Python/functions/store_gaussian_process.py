import pickle


def store_gaussian_process(gaussian_process, filename):
    path = '/home/samuel/Documents/EP_GP/GP_Python/saved_GPs/'
    file = open(path+filename, 'wb')
    pickle.dump(gaussian_process, file)
    file.close()
    pass


def load_gaussian_process(filename):
    path = '/home/samuel/Documents/EP_GP/GP_Python/saved_GPs/'
    file = open(path+filename, 'rb')
    gaussian_process = pickle.load(file)
    #print(GaussianProcess)
    file.close()
    return gaussian_process


