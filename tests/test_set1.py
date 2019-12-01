import pytest

from mypkg.price_range_prediction import *

# Testing if there are no non-relevant or no response entries left in the dataset
def test_case1():
    count = 0
    for i, j in data.iterrows(): 
        for item in j:
            if item in ['-6', -6, '-9', -9, 'M', 'N']: count+=1
    assert(count == 0)

    
# Testing if number of rows of dataset after data cleaning remains correct (input dataset always remains the same, so this test should 
# always satisfy
def test_case2():
    assert(len(clean_data.index) == 36358)
    
    
# Testing if number of columns of dataset after data cleaning remains correct (input dataset always remains the same, so this test should 
# always satisfy
def test_case3():
    assert(len(clean_data.columns) == 1007)
    
    
# Testing if all Nan/Null entries are replaced in final dataset
def test_case4():
    assert(clean_data.isnull().sum().sum() == 0)

