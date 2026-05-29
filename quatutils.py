import numpy as np

class Quaternion:

    def __init__(self, components):
        """Quaternion class constructor"""
        self.components = np.array(components)

    def __repr__(self):
        return f"Quaternion({self.components.tolist()})"

    @property
    def q0(self):
        return self.components[0]

    @property
    def q1(self):
        return self.components[1]

    @property
    def q2(self):
        return self.components[2]

    @property
    def q3(self):
        return self.components[3]

    def __add__(self, other):
        return Quaternion(self.components + other.components)

    def __sub__(self, other):
        return Quaternion(self.components - other.components)

    def __mul__(self, other):

        if isinstance(other, Quaternion):

            a1, b1, c1, d1 = self.components
            a2, b2, c2, d2 = other.components

            return Quaternion([
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
            ])

        elif np.isscalar(other):

            return Quaternion(self.components * other)

        return NotImplemented

    def __rmul__(self, other):
        return self * other

    @property
    def scalar(self):
        """
        Get scalar component of quaternion.
        Returns:
            Scalar: Scalar of quaternion
        """
        return self.q0

    @property
    def vector(self):
        """
        Get vector component of quaternion.
        Returns:
            Vector (x, y, z): Vector of quaternion
        """
        return np.array([self.q1, self.q2, self.q3])

    @property
    def conjugate(self):
        """
        Get conjugate of quaternion.
        Returns:
            Quaternion: Conjugate of quaternion
        """
        return Quaternion([self.q0, -self.q1, -self.q2, -self.q3])

    @property
    def norm(self):
        """
        Get norm of quaternion.
        Returns:
            Scalar: Norm of quaternion
        """
        return np.linalg.norm(self.components)

    @property
    def unit_quaternion(self):
        """
        Get unit quaternion.
        Returns:
            Quaternion: Unit quaternion
        """
        if self.norm == 0:
            return self
        return Quaternion(self.components / self.norm)

    def rotate(self, vector):
        """
        Rotate a vector using quaternion.
        Inputs:
            Vector (x,y,z): Vector to rotate
        Returns:
            Vector (x,y,z): Rotated vector
        """
        q = self.unit_quaternion
        q_vector = Quaternion([0, *vector])
        rotated = q * q_vector * q.conjugate
        return rotated.vector