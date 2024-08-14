import numpy as np
import random
from torchvision import transforms

def divide_image(image, n):
    h, w, _ = image.shape
    cell_h, cell_w = h // n, w // n
    cells = []

    for i in range(n):
        for j in range(n):
            cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cells.append(cell)
    
    return cells, cell_h, cell_w

def shuffle_cells(cells, n, k):
    # Shuffle indices of cells
    indices = list(range(n * n))
    random.shuffle(indices)

    # Select k pairs of indices to swap
    for i in range(k):
        idx1, idx2 = indices[i], indices[i + n * n // 2]
        cells[idx1], cells[idx2] = cells[idx2], cells[idx1]
    
    return cells

def reconstruct_image(cells, n, cell_h, cell_w):
    h, w, _ = cells[0].shape  # Get the shape of a single cell
    new_image = np.zeros((n * cell_h, n * cell_w, 3), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            new_image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = cells[i * n + j]
    
    return new_image

def custom_augmentation(image, n=3, k=3):
    cells, cell_h, cell_w = divide_image(image, n)
    shuffled_cells = shuffle_cells(cells, n, k)
    shuffled_image = reconstruct_image(shuffled_cells, n, cell_h, cell_w)
    return shuffled_image

class CustomTransform:
    def __call__(self, image):
        image = np.array(image)
        image = custom_augmentation(image)
        return transforms.ToTensor()(image)
