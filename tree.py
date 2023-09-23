import os

def print_directory_tree(path, prefix=''):
    if os.path.isdir(path):
        files = os.listdir(path)
        files.sort()
        for i, file in enumerate(files):
            if i == len(files) - 1:
                print(prefix + '└─── ' + file)
                print_directory_tree(path + '/' + file, prefix + '    ')
            else:
                print(prefix + '├─── ' + file)
                print_directory_tree(path + '/' + file, prefix + '│   ')

# print_directory_tree('.')
