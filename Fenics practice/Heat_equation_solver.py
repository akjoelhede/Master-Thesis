import numpy as np
import matplotlib.pyplot as plt
import fenics as fe



n_elements = 32
mesh = fe.UnitIntervalMesh(n_elements)

# define function space
lagrange_polynomial_space_first_order = fe.FunctionSpace(mesh, "Lagrange", 1)

# the value of the solutio on the boundary

u_on_boundary = fe.Constant(0.0)

# a function that says if we are on the boundary

def boolean_boundary(x,on_boundary):
    return on_boundary

# homogenous diriclet boundary conditions
boundary_condition = fe.DirichletBC(lagrange_polynomial_space_first_order, u_on_boundary, boolean_boundary)


# the initial condition, u(t=0, x) = sin(pi * x)

initial_condition = fe.Expression("sin(3.141 * x[0])", degree=1)

u_old = fe.interpolate(initial_condition, lagrange_polynomial_space_first_order)

plt.figure()
fe.plot(u_old, label="t=0.0")

# the time step, euler step
time_step_length = 0.1

#the forcing on the rhs of the PDE
heat_source = fe.Constant(0.0)

#create the FE problem

u_trial = fe.TrialFunction(lagrange_polynomial_space_first_order)
v_test = fe.TestFunction(lagrange_polynomial_space_first_order)

weak_form_residuum = (

    u_trial *v_test *fe.dx
    +
    time_step_length * fe.dot(
        fe.grad(u_trial),
        fe.grad(v_test)
    ) * fe.dx
    -
    (
        u_old * v_test *fe.dx
        +
        time_step_length *heat_source *v_test*fe.dx
    )
)

# linear PDE that is separable into lhs and rhs

weak_form_lhs = fe.lhs(weak_form_residuum)
weak_form_rhs = fe.rhs(weak_form_residuum)


u_solution = fe.Function(lagrange_polynomial_space_first_order)
#time stepping

n_time_steps = 5

time_current = 0.0

for i in range(n_time_steps):
    time_current += time_step_length

    # finite element assembly 

    fe.solve(
        weak_form_lhs == weak_form_rhs, u_solution, boundary_condition
    )

    u_old.assign(u_solution)

    fe.plot(u_solution, label = f"t= {time_current:1.1f}")

plt.legend()
plt.title("Heat conduction in a rod with homogenous diriclet boundary conditions")
plt.xlabel("x position")
plt.ylabel("Temperature")
plt.grid()
plt.show()
plt.savefig("Heat Conduction", dpi = 300)
