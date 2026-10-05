import numpy as np
import matplotlib.pyplot as plt
objective_coefficients = input(
    "Enter the coefficients for the objective function (comma-separated): "
).split(",")

objective_coefficients = [
    float(c.strip()) for c in objective_coefficients
]
num_constraints = int(
    input("Enter the number of constraints: ")
)

constraints = []
for i in range(1, num_constraints + 1):

    coefficients = input(
        f"Enter the coefficients for constraint {i} (comma-separated): "
    ).split(",")

    coefficients = [
        float(c.strip()) for c in coefficients
    ]

    rhs_value = float(
        input(
            f"Enter the right-hand side value for constraint {i}: "
        )
    )

    relation = input(
        f"Enter the relation (<=, >=, or =) for constraint {i}: "
    ).strip()

    constraints.append(
        (coefficients, rhs_value, relation)
    )

points = []
points.append((0, 0))
for coefficients, rhs, relation in constraints:

    a = coefficients[0]
    b = coefficients[1]
    if a != 0:
        x_value = rhs / a

        if x_value >= 0:
            points.append((x_value, 0))
    if b != 0:
        y_value = rhs / b

        if y_value >= 0:
            points.append((0, y_value))
for i in range(len(constraints)):

    for j in range(i + 1, len(constraints)):

        a1, b1 = constraints[i][0]
        c1 = constraints[i][1]

        a2, b2 = constraints[j][0]
        c2 = constraints[j][1]

        determinant = a1 * b2 - a2 * b1

        if determinant != 0:

            x_value = (
                c1 * b2 - c2 * b1
            ) / determinant

            y_value = (
                a1 * c2 - a2 * c1
            ) / determinant

            if x_value >= 0 and y_value >= 0:
                points.append(
                    (x_value, y_value)
                )

def is_feasible(x, y):

    for coefficients, rhs, relation in constraints:

        a = coefficients[0]
        b = coefficients[1]

        left_side = a * x + b * y

        if relation == "<=":

            if left_side > rhs + 1e-6:
                return False

        elif relation == ">=":

            if left_side < rhs - 1e-6:
                return False

        elif relation == "=":

            if abs(left_side - rhs) > 1e-6:
                return False

    return True
feasible_points = []

for x, y in points:

    if is_feasible(x, y):

        feasible_points.append((x, y))
unique_points = []

for point in feasible_points:

    if not any(
        np.allclose(point, p)
        for p in unique_points
    ):

        unique_points.append(point)
if len(unique_points) == 0:

    print("Status: Infeasible")

    plt.figure(figsize=(8, 6))

else:

    objective_values = []

    for x, y in unique_points:

        z = (
            objective_coefficients[0] * x
            + objective_coefficients[1] * y
        )

        objective_values.append(z)

    
    optimal_index = np.argmax(
        objective_values
    )

    x_optimal, y_optimal = unique_points[
        optimal_index
    ]

    optimal_z = objective_values[
        optimal_index
    ]

    print("Status: Optimal")
    print("Objective value (Z):", optimal_z)

    print("Optimal values:")
    print("x =", x_optimal)
    print("y =", y_optimal)



    plt.figure(figsize=(8, 6))

    x_values = np.linspace(0, 10, 400)


    for i, (
        coefficients,
        rhs,
        relation
    ) in enumerate(
        constraints,
        start=1
    ):

        a = coefficients[0]
        b = coefficients[1]

        if b != 0:

            y_values = (
                rhs - a * x_values
            ) / b

            plt.plot(
                x_values,
                y_values,
                label=f"Constraint {i}"
            )

        elif a != 0:

            x_intercept = rhs / a

            plt.axvline(
                x_intercept,
                linestyle="--",
                label=f"Constraint {i}"
            )


    
    if unique_points:

        x_points = [
            p[0] for p in unique_points
        ]

        y_points = [
            p[1] for p in unique_points
        ]

        plt.scatter(
            x_points,
            y_points,
            marker="o",
            s=60,
            label="Feasible Points"
        )

    plt.scatter(
        x_optimal,
        y_optimal,
        color="red",
        marker="*",
        s=200,
        label="Optimal Solution"
    )

    X, Y = np.meshgrid(
        np.linspace(0, 10, 100),
        np.linspace(0, 10, 100)
    )

    Z = (
        objective_coefficients[0] * X
        + objective_coefficients[1] * Y
    )

    plt.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap="coolwarm",
        alpha=0.3
    )

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    plt.title(
        "Linear Programming Solution"
    )

    plt.legend()
    plt.grid(True)

    plt.axhline(
        0,
        color="black",
        linewidth=0.5
    )

    plt.axvline(
        0,
        color="black",
        linewidth=0.5
    )

    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.show()
