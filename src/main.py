"""
Module: main.py
Description: This module contains the main function for the project.
"""

from argparse import ArgumentParser

from src.model import Model
from src.data import read_data, write_data

parser = ArgumentParser(description="Linear Model")
parser.add_argument("--weight", type=float, default=2.0, help="Weight of the model")
parser.add_argument("--intercept", type=float, default=0.0, help="Intercept of the model")
parser.add_argument("--input", type=str, help="Path to the input data")
parser.add_argument("--output", type=str, help="Path to the output data")
args = parser.parse_args()

weight = args.weight
intercept = args.intercept
input_path = args.input
output_path = args.output

model = Model(weight, intercept)

input_data = read_data(input_path)
output_data = model.predict(input_data)
write_data(output_path, output_data)

print(output_data)
print("Successfully wrote output data to", output_path)
