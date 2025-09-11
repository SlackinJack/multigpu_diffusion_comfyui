import os
from glob import glob


def getModelFilesInFolder(folder_in):
    return __get_files_in_folder(folder_in, ["safetensors"])


def getModelFilesInFolderUnsafe(folder_in):
    return __get_files_in_folder(folder_in, ["safetensors", "bin", "ckpt", "pth"])


def getModelFilesInFolderGGUF(folder_in):
    return __get_files_in_folder(folder_in, ["gguf"])


def getModelSubfoldersInFolder(folder_in):
    return __get_folders_in_folder(folder_in)


def __get_folders_in_folder(folder_in):
    out = []
    for path in os.listdir(folder_in):
        if os.path.isdir(os.path.join(folder_in, path)):
            out.append(path)
    return out


def __get_files_in_folder(folder_in, model_extensions):
    out = []
    for path in os.walk(folder_in):
        for f in glob(os.path.join(path[0], '*.*'), recursive=True):
            if f.split(".")[-1] in model_extensions:
                out.append(f.replace(folder_in + "/", ""))
    return out
