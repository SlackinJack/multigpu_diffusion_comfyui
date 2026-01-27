import os
from glob import glob


def getModelFilesInFolder(folder_in, blacklist_folders=[]):
    return __get_files_in_folder(folder_in, ["safetensors", "sft"], blacklist_folders)


def getModelFilesInFolderUnsafe(folder_in, blacklist_folders=[]):
    return __get_files_in_folder(folder_in, ["safetensors", "sft", "bin", "ckpt", "pth", "gguf"], blacklist_folders)


def getModelConfigsInFolder(folder_in, blacklist_folders=[]):
    return __get_files_in_folder(folder_in, ["json"], blacklist_folders)


def getModelSubfoldersInFolder(folder_in, blacklist_folders=[]):
    return __get_folders_in_folder(folder_in, blacklist_folders)


def __get_files_in_folder(folder_in, file_exts, blacklist_folders):
    out = []
    for root, dirs, files in os.walk(folder_in, followlinks=True):
        dirs[:] = [d for d in dirs if d not in blacklist_folders]
        for f in files:
            if any(f.endswith(ext) for ext in file_exts):
                out.append(os.path.relpath(os.path.join(root, f), folder_in))
    return out


def __get_folders_in_folder(folder_in, blacklist_folders):
    out = []
    if os.path.isdir(folder_in):
        for root, dirs, _ in os.walk(folder_in, followlinks=True):
            dirs[:] = [d for d in dirs if d not in blacklist_folders]
            for d in dirs:
                out.append(os.path.relpath(os.path.join(root, d), folder_in))
    return out
