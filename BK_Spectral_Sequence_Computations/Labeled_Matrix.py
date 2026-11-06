import numpy as np
import pandas as pd
from scipy.linalg import null_space
from sympy import Matrix
from sympy.polys.matrices import DomainMatrix
from typing import List, Union
from time import time

class LabeledMatrix:
    def __init__(self, data: Union[np.ndarray, List[List[float]]],
                 row_labels, col_labels):
        """
        A matrix whose rows and columns are indexed by elements of given sets.

        :param data: 2D list or numpy array containing numerical data.
        :param row_labels: List of unique labels for the rows.
        :param col_labels: List of unique labels for the columns.
        """
        # Keep integer matrices integral: the differentials in Complex have
        # integer coefficients and their rational rank must be computed exactly.
        self.data = np.asarray(data).copy()
        self.row_labels = list(row_labels)
        self.col_labels = list(col_labels)

        if len(set(self.row_labels)) != len(self.row_labels) or len(set(self.col_labels)) != len(self.col_labels):
            raise ValueError("Row and column labels must be unique.")
        self.row_index = {label: i for i, label in enumerate(self.row_labels)}
        self.col_index = {label: j for j, label in enumerate(self.col_labels)}

        if self.data.shape != (len(self.row_labels), len(self.col_labels)):
            raise ValueError("Data dimensions do not match provided labels.")

    def __repr__(self):
        """Pretty print the matrix as a pandas DataFrame."""
        return str(pd.DataFrame(self.data, index=self.row_labels, columns=self.col_labels))

    def __getitem__(self, key):
        """Allows accessing elements using (row_label, col_label)."""
        row_label, col_label = key[0], key[1]
        return self.data[self.row_index[row_label], self.col_index[col_label]]

    def __setitem__(self, key, value):
        """Allows setting elements using (row_label, col_label)."""
        row_label, col_label = key[0], key[1]
        self.data[self.row_index[row_label], self.col_index[col_label]] = value

    def transpose(self):
        """Returns the transpose of the matrix."""
        return LabeledMatrix(self.data.T, self.col_labels, self.row_labels)

    def __add__(self, other):
        """Element-wise addition of two matrices with the same labels."""
        if not isinstance(other, LabeledMatrix):
            raise TypeError("Can only add another LabeledMatrix.")
        if self.row_labels != other.row_labels or self.col_labels != other.col_labels:
            raise ValueError("Matrices must have the same labels.")
        return LabeledMatrix(self.data + other.data, self.row_labels, self.col_labels)

    def __sub__(self, other):
        """Element-wise subtraction."""
        return self + (-1 * other)

    def __mul__(self, other):
        """Matrix multiplication using label-aware dot product."""
        if isinstance(other, LabeledMatrix):
            if self.col_labels != other.row_labels:
                raise ValueError("Incompatible label sets for multiplication.")
            product = np.dot(self.data, other.data)
            return LabeledMatrix(product, self.row_labels, other.col_labels)
        elif isinstance(other, (int, float, np.integer)):
            return LabeledMatrix(self.data * other, self.row_labels, self.col_labels)
        else:
            raise TypeError("Unsupported multiplication type.")

    def inverse(self):
        """Computes the inverse of the matrix if it is square and invertible."""
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Matrix must be square to be inverted.")
        return LabeledMatrix(np.linalg.inv(self.data.astype(float)), self.row_labels, self.col_labels)

    def to_dataframe(self):
        """Converts the labeled matrix to a Pandas DataFrame."""
        return pd.DataFrame(self.data, index=self.row_labels, columns=self.col_labels)

    def rank(self, exact=True):
        """Rank over Q for integral matrices (exact=True), or numerical rank."""
        if not self.row_labels or not self.col_labels:
            return 0
        if not exact:
            return int(np.linalg.matrix_rank(self.data.astype(float)))
        if not all(isinstance(x, (int, np.integer)) for x in self.data.flat):
            raise TypeError("Exact rank is available only for integer matrices.")
        # DomainMatrix uses exact arithmetic over ZZ/QQ (no SVD tolerance).
        return int(DomainMatrix.from_Matrix(Matrix(self.data.tolist())).rank())

    def kernel(self):
        """Numerical basis of the kernel (not suitable for exact homology)."""
        kernel = null_space(self.data.astype(float))
        col_labels = [('Lin Comb', i) for i in range(kernel.shape[1])]
        return LabeledMatrix(kernel, self.col_labels, col_labels)

    def round(self, threshold):
        self.data = np.where(np.abs(self.data) < threshold, 0.0, self.data)


if __name__ == "__main__":
    mat = LabeledMatrix([[1, 2], [2, 4]], row_labels=["a", "b"], col_labels=["x", "y"])
    print("Matrix:")
    print(mat)
    print("Transpose:")
    print(mat.transpose())
    print("Element Access (\"a\", \"x\"):", mat["a", "x"])
    print(mat.kernel())

    # Benchmark only when running this module directly (not on import).
    A = np.random.rand(2000, 2000)
    begin = time()
    print("Rank of the matrix:", np.linalg.matrix_rank(A))
    print(time()-begin, ' seconds')
