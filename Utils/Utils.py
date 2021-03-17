import os

def createFolders(folder):

    if not os.path.exists(folder):
        os.mkdir(folder)