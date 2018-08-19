import math


def next_batch(X, y, batch_size):
    # Step 1.0: Calculate batches count
    batch_count = int(math.ceil(len(X) / batch_size))
    # Step 1.1: Generate the next batch
    for curr in range(batch_count):
        batch_beginning = curr
        batch_end = curr + min(batch_size, len(X) - curr * batch_size)
        yield X[batch_beginning: batch_end, :], y[batch_beginning: batch_end]
