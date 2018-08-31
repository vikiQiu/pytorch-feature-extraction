import os


def check_dir_exists(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
