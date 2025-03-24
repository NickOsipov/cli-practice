"""
Module: main.py
Description: This module contains the main function for the project.
"""

from argparse import ArgumentParser, Namespace
import os
import sys

from loguru import logger

from src.model import Model
from src.data import read_data, write_data

LOGS_PATH = os.path.join("logs", "main.log")
logger.add(LOGS_PATH, level="DEBUG", rotation="10 MB")


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
    logger.info("Starting the main function.")

    args = parse_args()

    weight = args.weight
    intercept = args.intercept
    input_path = args.input
    output_path = args.output

    logger.debug(f"Weight: {weight}")
    logger.debug(f"Intercept: {intercept}")
    logger.debug(f"Input path: {input_path}")
    logger.debug(f"Output path: {output_path}")

    logger.info("Creating the model.")
    model = Model(weight, intercept)
    logger.info("Model created successfully.")

    try:
        logger.info(f"Reading input data from {input_path}")
        input_data = read_data(input_path)

        logger.info("Predicting output data.")
        output_data = model.predict(input_data)
        logger.success("Prediction completed successfully.")

        write_data(output_path, output_data)
        logger.success(f"Successfully wrote output data to {output_path}")

    except FileNotFoundError:
        logger.error(f"Input file {input_path} not found.")
        sys.exit(1)

    except Exception as e: # pylint: disable=broad-except
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
