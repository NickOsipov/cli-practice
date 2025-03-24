"""
Module: test_model.py
Description: This module contains the tests for the model module.
"""

from typing import List

import pytest

from src.model import Model

@pytest.mark.parametrize("weight, intercept, data, expected", [
    (2.0, 0.0, [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]),
    (3.0, 1.0, [1.0, 2.0, 3.0, 4.0], [4.0, 7.0, 10.0, 13.0]),
    (4.0, 2.0, [1.0, 2.0, 3.0, 4.0], [6.0, 10.0, 14.0, 18.0])
])
def test_predict(weight: float, intercept: float, data: List[float], expected: List[float]) -> None:
    """
    Test the model

    Parameters
    ----------
    weight : float
        Weight of the model
    intercept : float
        Intercept of the model
    data : List[float]
        Input data
    expected : List[float]
        Expected result
    """

    model = Model(weight, intercept)
    assert model.predict(data) == expected
