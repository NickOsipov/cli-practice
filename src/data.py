"""
Module: data.py
Description: This module contains functions for reading and writing data.
"""

from typing import List
import csv


def read_data(file_path: str) -> List[float]:
    """
    Read data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    List[float]
        List of data read from the CSV file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        result = []
        for row in reader:
            try:
                result.append(float(row[0]))
            except (ValueError, IndexError):
                continue
    return result

def write_data(file_path: str, data: List[float]) -> None:
    """
    Write data to a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    data : List[float]
        List of data to write to the CSV file.
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for value in data:
            writer.writerow([value])
