"""
Module: main.py
Description: This module contains the main function for the project.
"""

from argparse import ArgumentParser, Namespace
import sys

from src.model import Model
from src.data import read_data, write_data


def parse_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description="Linear Model")
    parser.add_argument("--weight", type=float, default=2.0, help="Weight of the model")
    parser.add_argument("--intercept", type=float, default=0.0, help="Intercept of the model")
    parser.add_argument("--input", type=str, help="Path to the input data")
    parser.add_argument("--output", type=str, help="Path to the output data")

    return parser.parse_args()


def main() -> None:
    """
    Main function of the project.
    """
    args = parse_args()

    weight = args.weight
    intercept = args.intercept
    input_path = args.input
    output_path = args.output

    model = Model(weight, intercept)

    try:
        input_data = read_data(input_path)
        output_data = model.predict(input_data)
        write_data(output_path, output_data)

        print(output_data)
        print("Successfully wrote output data to", output_path)

    except FileNotFoundError:
        print("Input file not found.")
        sys.exit(1)

    except Exception as e: # pylint: disable=broad-except
        print("An error occurred:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
