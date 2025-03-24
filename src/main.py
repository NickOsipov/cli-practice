"""
Module: main.py
Description: This module contains the main function for the project.
"""

from argparse import ArgumentParser

from src.model import model

parser = ArgumentParser(description="Linear Model")
parser.add_argument("--weight", type=float, default=2.0, help="Weight of the model")
parser.add_argument("--intercept", type=float, default=0.0, help="Intercept of the model")
args = parser.parse_args()

input_data = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
output_data = [model(x, args.weight, args.intercept) for x in input_data]

print(output_data)