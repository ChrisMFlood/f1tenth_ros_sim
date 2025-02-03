# Code Citations

## License: MIT
https://github.com/f1tenth/f1tenth_lab9_template/tree/752cd1df490adf249cfd3f88d9ac5475c72655f6/mpc/scripts/mpc_node.py

```
def mpc_prob_init(self):
    """
    Create MPC quadratic optimization problem using cvxpy, solver: OSQP
    Will be solved every iteration for control.
    More MPC problem info1334rmation here: https://osqp.org/docs/examples/mpc.html
    More QP example in CVXPY here
```


## License: Apache_2_0
https://github.com/f1tenth-class/safe-MPC-blocking/tree/b5f56e54190cf84d0e4db5b66830221cd0678a54/mpc/scripts/mpc_node.py

```
self):
    """
    Create MPC quadratic optimization problem using cvxpy, solver: OSQP
    Will be solved every iteration for control.
    More MPC problem information here: https://osqp.org/docs/examples/mpc.html
    More QP example in CVXPY here: https:
```


## License: MIT
https://github.com/derekhanbaliq/f1tenth-software-stack/tree/76361f603871fd819a496c4bcc36bcf3b6718d83/mpc/scripts/mpc_node.py

```
(Q_block))  # (4 * 9) x (4 * 9), Qk + Qfk

    # Formulate and create the finite-horizon optimal control problem (objective function)
    # The FTOCP has the horizon of T timesteps

    # --------------------------------------------------------
    # TODO: fill in the objectives here,
```


## License: MIT
https://github.com/Damowerko/f1ten-tigers/tree/9898e9625491a4accf6a9bb14f54f6c278fa5948/tigerstack/tigerstack/mpc_node.py

```
vec(self.uk), R_block)

    # Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
    objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
```


## License: MIT
https://github.com/Damowerko/f1ten-tigers/tree/9898e9625491a4accf6a9bb14f54f6c278fa5948/tigerstack/tigerstack/mpc/mpc_node.py

```
uk), R_block)

    # Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
    objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

    # Objective part 3
```

