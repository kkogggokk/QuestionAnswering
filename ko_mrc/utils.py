from typing import List, Dict, Any, Sequence
from soynlp.hangle import jamo_levenshtein
import pandas as pd
import numpy as np


BASE_DATA = pd.read_csv("Your Path")


def edit_distance(test_data):
    BASE_DATA.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
    result = []
    for i in range(len(BASE_DATA)):
        dist = jamo_levenshtein(BASE_DATA['Predicted'][i], test_data['Predicted'][i])
        result.append(dist)
    return np.mean(result) 

