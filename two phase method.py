import numpy as np


# ============================================================
# TWO-PHASE METHOD
# Pure Python + NumPy
# No SciPy
# No PuLP
# ============================================================

EPS = 1e-9
MAX_ITERATIONS = 100


# ============================================================
# DISPLAY TABLEAU
# ============================================================

def display_tableau(tableau, column_names, basis, iteration, phase):

    print("\n" + "=" * 80)
    print(f"PHASE {phase} - ITERATION {iteration}")
    print("=" * 80)

    print(
        f"{'Basis':>12}",
        end="\t"
    )

    for name in column_names:
        print(
            f"{name:>12}",
            end="\t"
        )

    print(f"{'RHS':>12}")
    print("-" * 80)

    for i in range(len(tableau) - 1):

        print(
            f"{basis[i]:>12}",
            end="\t"
        )

        for value in tableau[i]:

            if abs(value) < EPS:
                value = 0

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
            value = 0

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

    pivot_element = tableau[
        pivot_row,
        pivot_col
    ]

    if abs(pivot_element) < EPS:
        return False

    # Divide pivot row
    tableau[pivot_row] = (
        tableau[pivot_row] / pivot_element
    )

    # Make other values in pivot column zero
    for i in range(len(tableau)):

        if i != pivot_row:

            factor = tableau[
                i,
                pivot_col
            ]

            if abs(factor) > EPS:

                tableau[i] = (
                    tableau[i]
                    - factor * tableau[pivot_row]
                )

    return True


# ============================================================
# SIMPLEX
# ============================================================

def simplex(
    tableau,
    column_names,
    basis,
    phase
):

    iteration = 0

    while iteration < MAX_ITERATIONS:

        iteration += 1

        display_tableau(
            tableau,
            column_names,
            basis,
            iteration - 1,
            phase
        )

        # ----------------------------------------------------
        # Find entering variable
        # ----------------------------------------------------

        objective_row = tableau[-1]

        coefficients = objective_row[:-1]

        if np.min(coefficients) >= -EPS:

            print(
                "\nOptimality condition reached."
            )

            return True

        entering_col = int(
            np.argmin(coefficients)
        )

        entering_variable = (
            column_names[entering_col]
        )

        # ----------------------------------------------------
        # Ratio test
        # ----------------------------------------------------

        ratios = []

        for i in range(
            len(tableau) - 1
        ):

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

        # ----------------------------------------------------
        # Unbounded
        # ----------------------------------------------------

        if len(ratios) == 0:

            print(
                "\nProblem is UNBOUNDED."
            )

            return False

        # Minimum ratio
        ratios.sort(
            key=lambda x: x[0]
        )

        leaving_row = ratios[0][1]

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

        # ----------------------------------------------------
        # Pivot
        # ----------------------------------------------------

        pivot(
            tableau,
            leaving_row,
            entering_col
        )

        # Update basis
        basis[
            leaving_row
        ] = entering_variable

    print(
        "\nMaximum iterations reached."
    )

    return False


# ============================================================
# GET INPUT
# ============================================================

def get_input():

    print("\n" + "=" * 70)
    print("TWO-PHASE METHOD")
    print("=" * 70)

    # --------------------------------------------------------
    # Number of variables
    # --------------------------------------------------------

    while True:

        try:

            n = int(
                input(
                    "\nEnter the number of variables: "
                )
            )

            if n > 0:
                break

            print(
                "Enter a positive integer."
            )

        except ValueError:

            print(
                "Enter a valid integer."
            )

    # --------------------------------------------------------
    # Number of constraints
    # --------------------------------------------------------

    while True:

        try:

            m = int(
                input(
                    "Enter the number of constraints: "
                )
            )

            if m > 0:
                break

            print(
                "Enter a positive integer."
            )

        except ValueError:

            print(
                "Enter a valid integer."
            )

    # --------------------------------------------------------
    # Objective function
    # --------------------------------------------------------

    print(
        "\nEnter objective function coefficients:"
    )

    c = []

    for i in range(n):

        while True:

            try:

                value = float(
                    input(
                        f"c[{i + 1}] = "
                    )
                )

                c.append(value)

                break

            except ValueError:

                print(
                    "Enter a valid number."
                )

    c = np.array(c)

    # --------------------------------------------------------
    # Constraint matrix
    # --------------------------------------------------------

    print(
        "\nEnter constraint coefficients:"
    )

    A = []

    for i in range(m):

        row = []

        print(
            f"\nConstraint {i + 1}:"
        )

        for j in range(n):

            while True:

                try:

                    value = float(
                        input(
                            f"A[{i + 1}][{j + 1}] = "
                        )
                    )

                    row.append(value)

                    break

                except ValueError:

                    print(
                        "Enter a valid number."
                    )

        A.append(row)

    A = np.array(A)

    # --------------------------------------------------------
    # RHS
    # --------------------------------------------------------

    print(
        "\nEnter right-hand side values:"
    )

    b = []

    for i in range(m):

        while True:

            try:

                value = float(
                    input(
                        f"b[{i + 1}] = "
                    )
                )

                b.append(value)

                break

            except ValueError:

                print(
                    "Enter a valid number."
                )

    b = np.array(b)

    return c, A, b


# ============================================================
# MAIN TWO-PHASE METHOD
# ============================================================

def two_phase_method(c, A, b):

    m, n = A.shape

    # --------------------------------------------------------
    # Make RHS positive
    # --------------------------------------------------------

    for i in range(m):

        if b[i] < 0:

            A[i] = -A[i]
            b[i] = -b[i]

    # --------------------------------------------------------
    # Add artificial variables
    # --------------------------------------------------------

    artificial_variables = [
        f"a{i + 1}"
        for i in range(m)
    ]

    column_names = [
        f"x{i + 1}"
        for i in range(n)
    ]

    column_names.extend(
        artificial_variables
    )

    # A + Identity matrix
    A_extended = np.hstack(
        [
            A,
            np.eye(m)
        ]
    )

    # --------------------------------------------------------
    # Create Phase 1 tableau
    # --------------------------------------------------------

    tableau = np.zeros(
        (
            m + 1,
            n + m + 1
        )
    )

    tableau[
        :m,
        :n + m
    ] = A_extended

    tableau[
        :m,
        -1
    ] = b

    # Phase 1 objective:
    # Minimize sum of artificial variables.
    #
    # Tableau uses maximization form,
    # so objective row initially has -1.

    tableau[
        -1,
        n:n + m
    ] = -1

    # Initial basis = artificial variables

    basis = artificial_variables.copy()

    # --------------------------------------------------------
    # Make Phase 1 objective row canonical
    # --------------------------------------------------------

    for i in range(m):

        tableau[-1] = (
            tableau[-1]
            + tableau[i]
        )

    # --------------------------------------------------------
    # PHASE 1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1")
    print("=" * 70)

    print(
        "\nFinding a feasible solution..."
    )

    success = simplex(
        tableau,
        column_names,
        basis,
        1
    )

    if not success:

        raise Exception(
            "Phase 1 failed."
        )

    # --------------------------------------------------------
    # Check Phase 1 objective
    # --------------------------------------------------------

    phase1_value = tableau[
        -1,
        -1
    ]

    # Because tableau is maximization form,
    # feasible solution should give zero.

    if abs(phase1_value) > 1e-7:

        raise Exception(
            "Problem is INFEASIBLE."
        )

    print(
        "\nPhase 1 completed successfully."
    )

    print(
        "Feasible solution found."
    )

    # --------------------------------------------------------
    # Remove artificial variables
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REMOVING ARTIFICIAL VARIABLES")
    print("=" * 70)

    # Try to remove artificial variables from basis
    for i in range(m):

        if basis[i].startswith("a"):

            # Find a non-artificial column
            for j in range(n):

                if abs(
                    tableau[i, j]
                ) > EPS:

                    # Check whether pivoting
                    # can remove artificial variable
                    if all(
                        abs(
                            tableau[k, j]
                        ) < EPS
                        for k in range(m)
                        if k != i
                    ):

                        pivot(
                            tableau,
                            i,
                            j
                        )

                        basis[i] = (
                            column_names[j]
                        )

                        break

    # --------------------------------------------------------
    # PHASE 2
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2")
    print("=" * 70)

    # Create Phase 2 objective row
    phase2_objective = np.zeros(
        n + m + 1
    )

    # Maximization:
    # Z row = -C
    phase2_objective[
        :n
    ] = -c

    tableau[-1] = (
        phase2_objective
    )

    # Make objective row canonical
    for i in range(m):

        basic_variable = basis[i]

        if basic_variable.startswith("x"):

            variable_number = int(
                basic_variable[1:]
            )

            column = variable_number - 1

            coefficient = tableau[
                -1,
                column
            ]

            if abs(coefficient) > EPS:

                tableau[-1] = (
                    tableau[-1]
                    - coefficient
                    * tableau[i]
                )

    # --------------------------------------------------------
    # Phase 2 Simplex
    # --------------------------------------------------------

    success = simplex(
        tableau,
        column_names,
        basis,
        2
    )

    if not success:

        raise Exception(
            "Phase 2 failed. "
            "Problem may be unbounded."
        )

    # --------------------------------------------------------
    # Extract solution
    # --------------------------------------------------------

    solution = np.zeros(n)

    for i in range(m):

        basic_variable = basis[i]

        if basic_variable.startswith("x"):

            variable_number = int(
                basic_variable[1:]
            )

            if (
                1
                <= variable_number
                <= n
            ):

                solution[
                    variable_number - 1
                ] = tableau[
                    i,
                    -1
                ]

    # --------------------------------------------------------
    # Calculate objective value
    # --------------------------------------------------------

    optimal_value = np.dot(
        c,
        solution
    )

    return solution, optimal_value


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    try:

        c, A, b = get_input()

        solution, optimal_value = (
            two_phase_method(
                c,
                A,
                b
            )
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        print(
            "\nOptimal Solution:"
        )

        for i, value in enumerate(
            solution
        ):

            print(
                f"x{i + 1} = "
                f"{value:.6f}"
            )

        print(
            f"\nOptimal Value (Z) = "
            f"{optimal_value:.6f}"
        )

        print(
            "\nTwo-Phase Method completed successfully."
        )

        print("=" * 70)

    except Exception as e:

        print(
            "\nError:",
            e
        )
