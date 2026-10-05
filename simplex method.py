import numpy as np
from fractions import Fraction

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


EPSILON = 1e-9
BIG_M = 1000000
MAX_ITERATIONS = 100


def read_coefficients(prompt, count):
    """Read coefficients separated by spaces."""

    while True:
        try:
            values = input(prompt).strip().split()

            if len(values) != count:
                print(f"Please enter exactly {count} values.")
                continue

            return [
                float(Fraction(value))
                for value in values
            ]

        except ValueError:
            print("Invalid input. Use numbers or fractions.")


def print_tableau(
    tableau,
    basis,
    column_names,
    iteration,
    decimals
):
    """Display simplex tableau."""

    print("\n" + "=" * 70)
    print(f"TABLEAU {iteration}")
    print("=" * 70)

    row_names = basis + ["Z"]

    if PANDAS_AVAILABLE:

        data = np.round(
            tableau,
            decimals
        )

        df = pd.DataFrame(
            data,
            columns=column_names + ["RHS"],
            index=row_names
        )

        print(df.to_string())

    else:

        print(
            "Basis\t" +
            "\t".join(column_names) +
            "\tRHS"
        )

        for i, row in enumerate(tableau):

            values = "\t".join(
                f"{value:.{decimals}f}"
                for value in row
            )

            print(
                f"{row_names[i]}\t{values}"
            )


def build_tableau(
    objective,
    A,
    b,
    relations,
    variable_names,
    problem_type
):
    """Build the initial simplex tableau."""

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    objective = np.array(
        objective,
        dtype=float
    )

    rows = []
    processed_relations = []

    basis = []
    extra_columns = []
    artificial_columns = []

    # ---------------------------------------------
    # Process constraints
    # ---------------------------------------------

    for i in range(len(b)):

        row = A[i].copy()
        rhs = b[i]
        relation = relations[i]

        # Make RHS positive
        if rhs < 0:

            row = -row
            rhs = -rhs

            if relation == "<=":
                relation = ">="

            elif relation == ">=":
                relation = "<="

        rows.append(row)
        b[i] = rhs
        processed_relations.append(
            relation
        )

    # ---------------------------------------------
    # Add slack, surplus and artificial variables
    # ---------------------------------------------

    for i, relation in enumerate(
        processed_relations
    ):

        if relation == "<=":

            slack_name = "S" + str(i + 1)

            extra_columns.append(
                (
                    slack_name,
                    i,
                    1
                )
            )

            basis.append(slack_name)

        elif relation == ">=":

            surplus_name = "E" + str(i + 1)

            extra_columns.append(
                (
                    surplus_name,
                    i,
                    -1
                )
            )

            artificial_name = "A" + str(i + 1)

            extra_columns.append(
                (
                    artificial_name,
                    i,
                    1
                )
            )

            artificial_columns.append(
                artificial_name
            )

            basis.append(
                artificial_name
            )

        elif relation == "=":

            artificial_name = "A" + str(i + 1)

            extra_columns.append(
                (
                    artificial_name,
                    i,
                    1
                )
            )

            artificial_columns.append(
                artificial_name
            )

            basis.append(
                artificial_name
            )

        else:

            raise ValueError(
                "Constraint must be <=, >= or ="
            )

    # ---------------------------------------------
    # Column names
    # ---------------------------------------------

    column_names = variable_names.copy()

    for name, _, _ in extra_columns:
        column_names.append(name)

    total_columns = len(column_names)

    # ---------------------------------------------
    # Create tableau
    # ---------------------------------------------

    tableau = np.zeros(
        (
            len(rows) + 1,
            total_columns + 1
        )
    )

    # Original variable coefficients
    for i in range(len(rows)):

        tableau[
            i,
            :len(variable_names)
        ] = rows[i]

    # Slack / surplus / artificial variables
    for name, row_index, coefficient in extra_columns:

        column_index = column_names.index(
            name
        )

        tableau[
            row_index,
            column_index
        ] = coefficient

    # RHS
    for i in range(len(rows)):

        tableau[
            i,
            -1
        ] = b[i]

    # ---------------------------------------------
    # Objective function
    # ---------------------------------------------

    if problem_type == 1:

        # Maximization
        max_objective = objective.copy()

    else:

        # Minimization converted to maximization
        max_objective = -objective.copy()

    tableau[
        -1,
        :len(variable_names)
    ] = -max_objective

    # ---------------------------------------------
    # Big-M penalty
    # ---------------------------------------------

    for artificial_name in artificial_columns:

        column_index = column_names.index(
            artificial_name
        )

        tableau[
            -1,
            column_index
        ] = BIG_M

    # ---------------------------------------------
    # Make objective row canonical
    # ---------------------------------------------

    for i, basic_variable in enumerate(basis):

        if basic_variable in artificial_columns:

            tableau[-1] -= (
                BIG_M * tableau[i]
            )

    return (
        tableau,
        basis,
        column_names,
        artificial_columns
    )


def simplex(
    tableau,
    basis,
    column_names,
    artificial_columns,
    decimals
):
    """Perform simplex iterations."""

    iteration = 1

    print_tableau(
        tableau,
        basis,
        column_names,
        iteration,
        decimals
    )

    while iteration < MAX_ITERATIONS:

        # -----------------------------------------
        # Find entering variable
        # -----------------------------------------

        objective_row = tableau[
            -1,
            :-1
        ]

        entering_index = np.argmin(
            objective_row
        )

        entering_value = objective_row[
            entering_index
        ]

        # Optimal solution reached
        if entering_value >= -EPSILON:
            break

        entering_variable = column_names[
            entering_index
        ]

        # -----------------------------------------
        # Ratio test
        # -----------------------------------------

        ratios = []

        for i in range(
            len(tableau) - 1
        ):

            coefficient = tableau[
                i,
                entering_index
            ]

            if coefficient > EPSILON:

                ratio = (
                    tableau[i, -1]
                    / coefficient
                )

                ratios.append(
                    (ratio, i)
                )

            else:

                ratios.append(
                    (np.inf, i)
                )

        # -----------------------------------------
        # Check unbounded solution
        # -----------------------------------------

        if all(
            ratio == np.inf
            for ratio, _ in ratios
        ):

            print(
                "\nProblem is UNBOUNDED."
            )

            return None

        # -----------------------------------------
        # Find pivot row
        # -----------------------------------------

        pivot_row = min(
            ratios,
            key=lambda x: x[0]
        )[1]

        pivot_element = tableau[
            pivot_row,
            entering_index
        ]

        leaving_variable = basis[
            pivot_row
        ]

        print(
            f"\nEntering variable : "
            f"{entering_variable}"
        )

        print(
            f"Leaving variable  : "
            f"{leaving_variable}"
        )

        print(
            f"Pivot element     : "
            f"{pivot_element:.{decimals}f}"
        )

        # -----------------------------------------
        # Normalize pivot row
        # -----------------------------------------

        tableau[
            pivot_row
        ] /= pivot_element

        # -----------------------------------------
        # Row operations
        # -----------------------------------------

        for i in range(
            len(tableau)
        ):

            if i != pivot_row:

                factor = tableau[
                    i,
                    entering_index
                ]

                tableau[i] -= (
                    factor
                    * tableau[pivot_row]
                )

        # -----------------------------------------
        # Update basis
        # -----------------------------------------

        basis[
            pivot_row
        ] = entering_variable

        iteration += 1

        print_tableau(
            tableau,
            basis,
            column_names,
            iteration,
            decimals
        )

    # ---------------------------------------------
    # Check maximum iterations
    # ---------------------------------------------

    if iteration >= MAX_ITERATIONS:

        print(
            "\nMaximum iterations reached."
        )

        return None

    # ---------------------------------------------
    # Check artificial variables
    # ---------------------------------------------

    for artificial_name in artificial_columns:

        artificial_index = column_names.index(
            artificial_name
        )

        for i, basic_variable in enumerate(
            basis
        ):

            if basic_variable == artificial_name:

                artificial_value = tableau[
                    i,
                    -1
                ]

                if artificial_value > EPSILON:

                    print(
                        "\nProblem is INFEASIBLE."
                    )

                    return None

    return tableau


def display_solution(
    tableau,
    basis,
    variable_names,
    original_objective,
    problem_type,
    decimals
):
    """Display final optimal solution."""

    solution = {
        variable: 0.0
        for variable in variable_names
    }

    # ---------------------------------------------
    # Extract variable values
    # ---------------------------------------------

    for i, basic_variable in enumerate(
        basis
    ):

        if basic_variable in variable_names:

            solution[
                basic_variable
            ] = tableau[
                i,
                -1
            ]

    print("\n" + "=" * 70)
    print("OPTIMAL SOLUTION")
    print("=" * 70)

    # ---------------------------------------------
    # Display variables
    # ---------------------------------------------

    for variable in variable_names:

        value = solution[
            variable
        ]

        if abs(value) < EPSILON:
            value = 0

        print(
            f"{variable} = "
            f"{value:.{decimals}f}"
        )

    # ---------------------------------------------
    # Calculate objective value
    # ---------------------------------------------

    values = np.array(
        [
            solution[var]
            for var in variable_names
        ]
    )

    objective_value = np.dot(
        original_objective,
        values
    )

    print(
        f"\nOptimal Z = "
        f"{objective_value:.{decimals}f}"
    )

    # ---------------------------------------------
    # Problem type
    # ---------------------------------------------

    if problem_type == 1:

        print(
            "Problem Type : Maximization"
        )

    else:

        print(
            "Problem Type : Minimization"
        )


def main():

    print("=" * 70)
    print("SIMPLEX METHOD")
    print("=" * 70)

    # ---------------------------------------------
    # Problem type
    # ---------------------------------------------

    while True:

        try:

            problem_type = int(
                input(
                    "\n"
                    "1 : Maximization\n"
                    "2 : Minimization\n"
                    "Enter problem type: "
                )
            )

            if problem_type in [1, 2]:
                break

            print(
                "Please enter 1 or 2."
            )

        except ValueError:

            print(
                "Invalid input."
            )

    # ---------------------------------------------
    # Number of variables
    # ---------------------------------------------

    while True:

        try:

            variable_count = int(
                input(
                    "\nEnter number of variables: "
                )
            )

            if variable_count > 0:
                break

            print(
                "Enter a positive number."
            )

        except ValueError:

            print(
                "Invalid input."
            )

    # ---------------------------------------------
    # Number of constraints
    # ---------------------------------------------

    while True:

        try:

            constraint_count = int(
                input(
                    "Enter number of constraints: "
                )
            )

            if constraint_count > 0:
                break

            print(
                "Enter a positive number."
            )

        except ValueError:

            print(
                "Invalid input."
            )

    # ---------------------------------------------
    # Variable names
    # ---------------------------------------------

    variable_names = [
        "X" + str(i + 1)
        for i in range(variable_count)
    ]

    print(
        "\nVariables:",
        ", ".join(variable_names)
    )

    # ---------------------------------------------
    # Objective function
    # ---------------------------------------------

    objective = read_coefficients(
        "\nEnter objective coefficients "
        "(space separated): ",
        variable_count
    )

    # ---------------------------------------------
    # Constraints
    # ---------------------------------------------

    A = []
    b = []
    relations = []

    print("\nEnter constraints.")

    print(
        "Example: 1 2 <= 8"
    )

    for i in range(
        constraint_count
    ):

        while True:

            try:

                user_input = input(
                    f"\nConstraint {i + 1}: "
                ).strip()

                parts = user_input.split()

                expected_parts = (
                    variable_count + 2
                )

                if len(parts) != expected_parts:

                    print(
                        f"Enter {variable_count} "
                        "coefficients, relation "
                        "and RHS."
                    )

                    continue

                # Coefficients
                coefficients = [
                    float(Fraction(value))
                    for value in parts[
                        :variable_count
                    ]
                ]

                # Relation
                relation = parts[
                    variable_count
                ]

                if relation not in [
                    "<=",
                    ">=",
                    "="
                ]:

                    print(
                        "Relation must be "
                        "<=, >= or =."
                    )

                    continue

                # RHS
                rhs = float(
                    Fraction(
                        parts[
                            variable_count + 1
                        ]
                    )
                )

                A.append(
                    coefficients
                )

                relations.append(
                    relation
                )

                b.append(
                    rhs
                )

                break

            except ValueError:

                print(
                    "Invalid input. "
                    "Please try again."
                )

    # ---------------------------------------------
    # Decimal places
    # ---------------------------------------------

    while True:

        try:

            decimals = int(
                input(
                    "\nNumber of decimal places "
                    "to display: "
                )
            )

            if decimals >= 0:
                break

            print(
                "Enter 0 or a positive number."
            )

        except ValueError:

            print(
                "Invalid input."
            )

    # ---------------------------------------------
    # Build initial tableau
    # ---------------------------------------------

    try:

        (
            tableau,
            basis,
            column_names,
            artificial_columns
        ) = build_tableau(
            objective,
            A,
            b,
            relations,
            variable_names,
            problem_type
        )

    except ValueError as error:

        print(
            "\nError:",
            error
        )

        return

    # ---------------------------------------------
    # Run Simplex
    # ---------------------------------------------

    final_tableau = simplex(
        tableau,
        basis,
        column_names,
        artificial_columns,
        decimals
    )

    if final_tableau is None:
        return

    # ---------------------------------------------
    # Display final solution
    # ---------------------------------------------

    display_solution(
        final_tableau,
        basis,
        variable_names,
        np.array(objective),
        problem_type,
        decimals
    )


if __name__ == "__main__":
    main()
