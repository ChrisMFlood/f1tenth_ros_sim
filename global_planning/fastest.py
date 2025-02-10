from sympy import *
import numpy
import pickle
import os


# Symbols and matrices definition
lr, L = symbols('l_r L')
dt, l, ds = symbols('Delta_t l Delta_s')
kappa = symbols('kappa')
s, n, mu = symbols('s n mu')
v, delta = symbols('v delta')

X = Matrix([n, mu])
U = Matrix([v, delta])

# Linearized state and control
s0, n0, mu0,  = symbols('s_0 n_0 mu_0')
v0, delta0 = symbols('v_0 delta_0')

Xs = Matrix([n0, mu0])
Us = Matrix([v0, delta0])


def state_function_t(X, U):
    beta = atan((lr / L) * tan(U[1]))
    vx = U[0] * cos(beta)
    vy = U[0] * sin(beta)
    phi_dot = U[0] / L * tan(U[1]) * cos(beta)
    A = Matrix([
        [(vx*sin(X[1]) - vy*cos(X[1]))],
        [phi_dot-(vx*cos(X[1]) - vy*sin(X[1]))/(1-X[0]*kappa)*kappa]])
    return simplify(A*(1/((vx*cos(X[1]) - vy*sin(X[1]))/(1-X[0]*kappa))))

def state_function(X, U):
    f = state_function_t(X, U)
    X_k = X + f * ds
    return X_k

# Convert a tensor to a matrix
def tensor_to_matrix(tensor):
    return Matrix([

        [tensor[0][0][0], tensor[0][1][0]],
        [tensor[0][1][0], tensor[1][1][0]]
        
    ])


def get_model_matrix():
    file_path = "/home/chris/sim_ws/src/global_planning/symbolic_matrices.pkl"

    if os.path.exists(file_path):
        # Load symbolic matrices if the file exists
        with open(file_path, "rb") as f:
            matrices = pickle.load(f)

        AA = matrices["AA"]
        BB = matrices["BB"]
        CC = matrices["CC"]

        print("Symbolic matrices loaded successfully!")
    else:
        # Calculate state and linearized form
        f = state_function(X, U)
        fs = state_function(Xs, Us)

        # Derivatives for linearization
        df_x = simplify(tensor_to_matrix(diff(fs, Xs.T)[0]))
        df_u = simplify(tensor_to_matrix(diff(fs, Us.T)[0]))

        # Linearized state function
        f_l = fs + df_x * (X - Xs) + df_u * (U - Us)
        f_l_simplified = simplify(f_l)

        AA = simplify(df_x)
        BB = simplify(df_u)
        CC = simplify(df_x * (-Xs) + df_u * (-Us) + fs)

        # Save symbolic matrices
        with open(file_path, "wb") as f:
            pickle.dump({"AA": AA, "BB": BB, "CC": CC}, f)

        print("Symbolic matrices saved successfully!")

    return AA, BB, CC


def main():
    A, B, C = get_model_matrix()


if __name__ == '__main__':
	main()