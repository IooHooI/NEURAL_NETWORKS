import math
import os


def next_batch(X, y, batch_size):
    # Step 1.0: Calculate batches count
    batch_count = int(math.ceil(len(X) / batch_size))
    # Step 1.1: Generate the next batch
    for curr in range(batch_count):
        batch_beginning = curr
        batch_end = curr + min(batch_size, len(X) - curr * batch_size)
        yield X[batch_beginning: batch_end, :], y[batch_beginning: batch_end]


def create_sub_folders(path):
    folders = path.split('/')
    sub_folder = ''
    for folder in folders:
        sub_folder += folder + '/'
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
