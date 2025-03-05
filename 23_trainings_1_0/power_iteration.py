import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    random_vector = np.random.rand(data.shape[0])
    random_vector = random_vector / np.linalg.norm(random_vector)
    for _ in range(num_steps):
        random_vector = np.dot(data, random_vector)
        random_vector = random_vector / np.linalg.norm(random_vector)
    return float(np.dot(random_vector, np.dot(data, random_vector)) / np.dot(random_vector, random_vector)), random_vector
