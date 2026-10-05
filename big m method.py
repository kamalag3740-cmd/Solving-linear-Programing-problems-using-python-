import numpy as np


# ============================================================
# BIG-M METHOD
# Pure Python + NumPy
# No PuLP / No external solver required
# ============================================================

M = 1000000.0
EPS = 1e-9
MAX_ITERATIONS = 100


# ============================================================
# DISPLAY TABLEAU
# ============================================================

def display_tableau(tableau, column_names, basis, iteration):
    print("\n" + "=" * 80)
    print(f"ITERATION {iteration}")
    print("=" * 80)

    header = ["Basis"] + column_names

    print("\t".join(f"{item:>12}" for item in header))
    print("-" * 80)

    for i in range(len(tableau) - 1):
        basis_name = basis[i]
        row = tableau[i]

        print(
            f"{basis_name:>12}",
            end="\t"
        )

        for value in row:
            if abs(value) < EPS:
                value = 0.0

            print(
                f"{value:>12.3f}",
                end="\t"
            )

        print()

    print("-" * 80)

    print(
        f"{'Z':>12}",
        end="\t"
    )

    for value in tableau[-1]:
        if abs(value) < EPS:
            value = 0.0

        print(
            f"{value:>12.3f}",
            end="\t"
        )

    print()
    print("=" * 80)


# ============================================================
# PIVOT OPERATION
# ============================================================

def pivot(tableau, pivot_row, pivot_col):

    pivot_element = tableau[pivot_row, pivot_col]

    if abs(pivot_element) < EPS:
        return False

    # Make pivot element 1
    tableau[pivot_row] = (
        tableau[pivot_row] / pivot_element
    )

    # Make all other values in pivot column 0
    for i in range(len(tableau)):

        if i != pivot_row:

            factor = tableau[i, pivot_col]

            if abs(factor) > EPS:

                tableau[i] = (
                    tableau[i]
                    - factor * tableau[pivot_row]
                )

    return True


# ============================================================
# GET ENTERING VARIABLE
# ============================================================

def get_entering_variable(objective_row):

    # Ignore RHS column
    coefficients = objective_row[:-1]

    min_value = np.min(coefficients)

    # For maximization:
    # Negative coefficient means entering variable
    if min_value >= -EPS:
        return None

    return int(
        np.argmin(coefficients)
    )


# ============================================================
# GET LEAVING VARIABLE
# ============================================================

def get_leaving_variable(
    tableau,
    entering_col
):

    ratios = []

    for i in range(len(tableau) - 1):

        coefficient = tableau[
            i,
            entering_col
        ]

        rhs = tableau[
            i,
            -1
        ]

        if coefficient > EPS:

            ratio = rhs / coefficient

            if ratio >= -EPS:
                ratios.append(
                    (ratio, i)
                )

    if not ratios:
        return None

    # Minimum positive ratio
    ratios.sort(
        key=lambda x: x[0]
    )

    return ratios[0][1]


# ============================================================
# INPUT NUMBER OF VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("BIG-M METHOD FOR LINEAR PROGRAMMING")
print("=" * 70)

while True:

    try:

        num_variables = int(
            input(
                "\nEnter the number of variables: "
            )
        )

        if num_variables > 0:
            break

        print(
            "Please enter a positive integer."
        )

    except ValueError:

        print(
            "Invalid input. Enter an integer."
        )


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

while True:

    try:

        objective_coefficients = input(
            "\nEnter objective function coefficients "
            "(comma-separated): "
        ).split(",")

        objective_coefficients = [
            float(x.strip())
            for x in objective_coefficients
        ]

        if len(objective_coefficients) != num_variables:

            print(
                f"Please enter exactly "
                f"{num_variables} coefficients."
            )

            continue

        break

    except ValueError:

        print(
            "Invalid input. Enter numbers only."
        )


# ============================================================
# NUMBER OF CONSTRAINTS
# ============================================================

while True:

    try:

        num_constraints = int(
            input(
                "\nEnter the number of constraints: "
            )
        )

        if num_constraints > 0:
            break

        print(
            "Please enter a positive integer."
        )

    except ValueError:

        print(
            "Invalid input. Enter an integer."
        )


# ============================================================
# READ CONSTRAINTS
# ============================================================

constraints = []

for i in range(1, num_constraints + 1):

    print(
        f"\n--- Constraint {i} ---"
    )

    while True:

        try:

            coefficients = input(
                "Enter coefficients "
                "(comma-separated): "
            ).split(",")

            coefficients = [
                float(x.strip())
                for x in coefficients
            ]

            if len(coefficients) != num_variables:

                print(
                    f"Please enter exactly "
                    f"{num_variables} coefficients."
                )

                continue

            break

        except ValueError:

            print(
                "Invalid input."
            )

    # Relation
    while True:

        relation = input(
            "Enter relation (<=, >= or =): "
        ).strip()

        if relation in ["<=", ">=", "="]:
            break

        print(
            "Please enter <=, >= or =."
        )

    # RHS
    while True:

        try:

            rhs = float(
                input(
                    "Enter right-hand side value: "
                )
            )

            break

        except ValueError:

            print(
                "Invalid number."
            )

    constraints.append(
        [
            coefficients,
            relation,
            rhs
        ]
    )


# ============================================================
# CONVERT NEGATIVE RHS
# ============================================================

processed_constraints = []

for coefficients, relation, rhs in constraints:

    coefficients = list(coefficients)

    if rhs < 0:

        coefficients = [
            -x for x in coefficients
        ]

        rhs = -rhs

        if relation == "<=":
            relation = ">="

        elif relation == ">=":
            relation = "<="

    processed_constraints.append(
        [
            coefficients,
            relation,
            rhs
        ]
    )


constraints = processed_constraints


# ============================================================
# CREATE COLUMN NAMES
# ============================================================

column_names = []

# Original variables
for i in range(num_variables):

    column_names.append(
        f"x{i + 1}"
    )


# Slack / Surplus / Artificial variables
slack_count = 0
surplus_count = 0
artificial_count = 0

constraint_variable_columns = []

for coefficients, relation, rhs in constraints:

    extra_variables = []

    if relation == "<=":

        slack_count += 1

        name = f"s{slack_count}"

        column_names.append(name)

        extra_variables.append(
            (name, 1)
        )

    elif relation == ">=":

        surplus_count += 1

        surplus_name = (
            f"e{surplus_count}"
        )

        column_names.append(
            surplus_name
        )

        extra_variables.append(
            (surplus_name, -1)
        )

        artificial_count += 1

        artificial_name = (
            f"a{artificial_count}"
        )

        column_names.append(
            artificial_name
        )

        extra_variables.append(
            (artificial_name, 1)
        )

    elif relation == "=":

        artificial_count += 1

        artificial_name = (
            f"a{artificial_count}"
        )

        column_names.append(
            artificial_name
        )

        extra_variables.append(
            (artificial_name, 1)
        )

    constraint_variable_columns.append(
        extra_variables
    )


# ============================================================
# CREATE TABLEAU
# ============================================================

num_columns = len(column_names)

tableau = np.zeros(
    (
        num_constraints + 1,
        num_columns + 1
    )
)

basis = []

artificial_columns = []


for i, (
    coefficients,
    relation,
    rhs
) in enumerate(constraints):

    # Original variables
    for j in range(num_variables):

        tableau[
            i,
            j
        ] = coefficients[j]

    # Extra variables
    for name, value in (
        constraint_variable_columns[i]
    ):

        col = column_names.index(name)

        tableau[
            i,
            col
        ] = value

        if name.startswith("a"):
            artificial_columns.append(col)

    # RHS
    tableau[
        i,
        -1
    ] = rhs

    # Determine initial basic variable

    if relation == "<=":

        basis.append(
            f"s{i + 1}"
        )

    elif relation == ">=":

        basis.append(
            f"a{sum(
                1
                for c in constraints[:i + 1]
                if c[1] in [">=", "="]
            )}"
        )

    elif relation == "=":

        basis.append(
            f"a{sum(
                1
                for c in constraints[:i + 1]
                if c[1] in [">=", "="]
            )}"
        )


# ============================================================
# CREATE OBJECTIVE ROW
# ============================================================

objective_row = np.zeros(
    num_columns + 1
)

# For maximization:
# Z row = -objective coefficients

for j in range(num_variables):

    objective_row[j] = (
        -objective_coefficients[j]
    )


# Big-M penalty for artificial variables
for col in artificial_columns:

    objective_row[col] = M


tableau[-1] = objective_row


# ============================================================
# MAKE OBJECTIVE ROW CANONICAL
# ============================================================

# If artificial variable is in the initial basis,
# subtract M times its row from objective row.

for i in range(num_constraints):

    basic_variable = basis[i]

    if basic_variable.startswith("a"):

        tableau[-1] = (
            tableau[-1]
            - M * tableau[i]
        )


# ============================================================
# DISPLAY INITIAL PROBLEM
# ============================================================

print("\n" + "=" * 70)
print("LINEAR PROGRAMMING PROBLEM")
print("=" * 70)

print("\nMaximize:")

objective_text = "Z = "

for i, coefficient in enumerate(
    objective_coefficients
):

    if i > 0 and coefficient >= 0:
        objective_text += " + "

    elif coefficient < 0:
        objective_text += " - "

    value = abs(coefficient)

    objective_text += (
        f"{value:g}x{i + 1}"
    )

print(objective_text)


print("\nSubject to:")

for coefficients, relation, rhs in constraints:

    expression = ""

    for i, coefficient in enumerate(
        coefficients
    ):

        if i > 0 and coefficient >= 0:
            expression += " + "

        elif coefficient < 0:
            expression += " - "

        value = abs(coefficient)

        expression += (
            f"{value:g}x{i + 1}"
        )

    print(
        f"{expression} {relation} {rhs:g}"
    )


print("\nAll variables >= 0")


# ============================================================
# SIMPLEX / BIG-M ITERATIONS
# ============================================================

iteration = 0

display_tableau(
    tableau,
    column_names + ["RHS"],
    basis,
    iteration
)


while iteration < MAX_ITERATIONS:

    iteration += 1

    # --------------------------------------------------------
    # Find entering variable
    # --------------------------------------------------------

    entering_col = get_entering_variable(
        tableau[-1]
    )

    # If no negative coefficient,
    # optimal solution reached.
    if entering_col is None:

        print(
            "\nOptimality condition satisfied."
        )

        break


    entering_variable = (
        column_names[entering_col]
    )


    # --------------------------------------------------------
    # Find leaving variable
    # --------------------------------------------------------

    leaving_row = get_leaving_variable(
        tableau,
        entering_col
    )

    if leaving_row is None:

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)

        print(
            "\nThe problem is UNBOUNDED."
        )

        break


    leaving_variable = basis[
        leaving_row
    ]

    pivot_element = tableau[
        leaving_row,
        entering_col
    ]


    print(
        f"\nEntering Variable : "
        f"{entering_variable}"
    )

    print(
        f"Leaving Variable  : "
        f"{leaving_variable}"
    )

    print(
        f"Pivot Element     : "
        f"{pivot_element:.6f}"
    )


    # --------------------------------------------------------
    # Pivot
    # --------------------------------------------------------

    success = pivot(
        tableau,
        leaving_row,
        entering_col
    )

    if not success:

        print(
            "\nPivot operation failed."
        )

        break


    # Update basis
    basis[
        leaving_row
    ] = entering_variable


    # --------------------------------------------------------
    # Display new tableau
    # --------------------------------------------------------

    display_tableau(
        tableau,
        column_names + ["RHS"],
        basis,
        iteration
    )


else:

    print(
        "\nMaximum iterations reached."
    )


# ============================================================
# CHECK ARTIFICIAL VARIABLES
# ============================================================

infeasible = False

for i in range(num_constraints):

    basic_variable = basis[i]

    if basic_variable.startswith("a"):

        rhs_value = tableau[
            i,
            -1
        ]

        if abs(rhs_value) > EPS:

            infeasible = True

            break


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)


if infeasible:

    print(
        "\nThe problem is INFEASIBLE."
    )

else:

    # --------------------------------------------------------
    # Extract variable values
    # --------------------------------------------------------

    solution = np.zeros(
        num_variables
    )

    for i in range(num_constraints):

        basic_variable = basis[i]

        if basic_variable.startswith("x"):

            variable_number = int(
                basic_variable[1:]
            )

            if 1 <= variable_number <= num_variables:

                solution[
                    variable_number - 1
                ] = tableau[
                    i,
                    -1
                ]


    # --------------------------------------------------------
    # Calculate original objective value
    # --------------------------------------------------------

    objective_value = np.dot(
        objective_coefficients,
        solution
    )


    print(
        "\nOptimal Solution:"
    )

    for i in range(num_variables):

        print(
            f"x{i + 1} = "
            f"{solution[i]:.6f}"
        )


    print(
        f"\nOptimal Z = "
        f"{objective_value:.6f}"
    )


    # --------------------------------------------------------
    # Constraint values
    # --------------------------------------------------------

    print(
        "\nConstraint Values:"
    )

    for index, (
        coefficients,
        relation,
        rhs
    ) in enumerate(constraints):

        lhs = np.dot(
            coefficients,
            solution
        )

        print(
            f"Constraint {index + 1}: "
            f"LHS = {lhs:.6f}, "
            f"{relation} RHS = {rhs:.6f}"
        )


print("\n" + "=" * 70)
print("BIG-M METHOD COMPLETED")
print("=" * 70)
