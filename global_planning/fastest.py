import math
import numpy as np 
def get_dynamic_model_matrix(self, delta, v, yaw, yawrate, beta, a, vehicle_params):
    """
    Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
    Linear System: Xdot = Ax +Bu + C
    State vector: x=[x, y, delta, v, yaw, yaw rate, beta]
    :param delta: steering angle
    :param v: speed
    :param phi: heading angle of the vehicle
    :return: A, B, C
    """
    # Extract Vehicle parameter to calculate vehicle dynamic model
    mass = 3.74 # mass in [kg]
    l_f = 0.15875  # Distance CoG to front in [m]
    l_r = 0.17145  # Distance CoG to back in [m]
    h_CoG = 0.074  # Height of the vehicle in [m]
    c_f = 4.718  # Cornering Stiffness front in [N]
    c_r = 5.4562  # Cornering Stiffness back in [N]
    Iz = 0.04712  # Vehicle Inertia [kg m2]
    mu = 1.0489  # friction coefficient [-]
    g = 9.81  # Vertical acceleration  [m/s2]
    # Calculate substitute/helper variables for all functions
    K = (mu * mass) / ((l_f + l_r) * Iz)
    T = (g * l_r) - (a * h_CoG)
    V = (g * l_f) + (a * h_CoG)
    F = l_f * c_f
    R = l_r * c_r
    M = (mu * c_f) / (l_f + l_r)
    N = (mu * c_r) / (l_f + l_r)
    A1 = K * F * T
    A2 = K * (R * V - F * T)
    A3 = K * (l_f * l_f * c_f * T + l_r * l_r * c_r * V)
    A4 = M * T
    A5 = N * V + M * T
    A6 = N * V * l_r - M * T * l_f
    B1 = (
        (-h_CoG * F * K) * delta
        + (h_CoG * K * (F + R)) * beta
        - (h_CoG * K * ((l_r * l_r * c_r) - (l_f * l_f * c_f))) * (yawrate / v)
    )
    B2 = (
        (-h_CoG * M) * (delta / v)
        - h_CoG * (N - M) * (beta / v)
        + h_CoG * (l_f * M + l_r * N) * (yawrate / (v * v))
    )
    # --------------  State (or system) matrix A, 7x7
    A = np.zeros((self.config.NX, self.config.NX))
    # Diagonal
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[4, 4] = 1.0
    A[5, 5] = -self.config.DT * (A3 / v) + 1
    A[6, 6] = -self.config.DT * A5 + 1
    # Zero row
    A[0, 3] = self.config.DT * math.cos(yaw + beta)
    A[0, 4] = -self.config.DT * v * math.sin(yaw + beta)
    A[0, 6] = -self.config.DT * v * math.sin(yaw + beta)
    # First Row
    A[1, 3] = self.config.DT * math.sin(yaw + beta)
    A[1, 4] = self.config.DT * v * math.cos(yaw + beta)
    A[1, 6] = self.config.DT * v * math.cos(yaw + beta)
    # Fourth Row
    A[4, 5] = self.config.DT
    # Fifth Row
    A[5, 2] = self.config.DT * A1
    A[5, 3] = self.config.DT * A3 * (yawrate / (v * v))
    A[5, 6] = self.config.DT * A2
    # Sixth Row
    A[6, 2] = self.config.DT * (A4 / v)
    A[6, 3] = (
        self.config.DT
        * (-A4 * beta * v + A5 * beta * v - A6 * 2 * yawrate)
        / (v * v * v)
    )
    A[6, 5] = self.config.DT * ((A6 / (v * v)) - 1)
    # -------------- Input Matrix B; 7x2
    B = np.zeros((self.config.NX, self.config.NU))
    B[2, 0] = self.config.DT
    B[3, 1] = self.config.DT
    B[5, 1] = self.config.DT * B1
    B[6, 1] = self.config.DT * B2
    # -------------- Matrix C; 7x1
    C = np.zeros(self.config.NX)
    C[0] = self.config.DT * (
        v * math.sin(yaw + beta) * yaw + v * math.sin(yaw + beta) * beta
    )
    C[1] = self.config.DT * (
        -v * math.cos(yaw + beta) * yaw - v * math.cos(yaw + beta) * beta
    )
    C[5] = self.config.DT * (-A3 * (yawrate / v) - B1 * a)
    C[6] = self.config.DT * (
        ((A4 * delta * v - A5 * beta * v + A6 * 2 * yawrate) / (v * v)) - B2 * a
    )
    return A, B, C

def get_kinematic_model_matrix(self, v, phi, delta):
    """
    Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
    Linear System: Xdot = Ax +Bu + C
    State vector: x=[x, y, v, yaw]
    :param v: speed
    :param phi: heading angle of the vehicle
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """
    # State (or system) matrix A, 4x4
    A = np.zeros((self.config.NXK, self.config.NXK))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = self.config.DTK * math.cos(phi)
    A[0, 3] = -self.config.DTK * v * math.sin(phi)
    A[1, 2] = self.config.DTK * math.sin(phi)
    A[1, 3] = self.config.DTK * v * math.cos(phi)
    A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB
    # Input Matrix B; 4x2
    B = np.zeros((self.config.NXK, self.config.NU))
    B[2, 0] = self.config.DTK
    B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)
    C = np.zeros(self.config.NXK)
    C[0] = self.config.DTK * v * math.sin(phi) * phi
    C[1] = -self.config.DTK * v * math.cos(phi) * phi
    C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)
    return A, B, C