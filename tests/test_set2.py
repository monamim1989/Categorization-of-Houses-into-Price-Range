import pytest

from mypkg.price_range_prediction import *

# Test if there are no Nan/Null entries still left in data_numeric
def test_case5():
    count = 0
    for i in numeric:
        for j in data_numeric[i]:
            if math.isnan(float(j)): 
                count += 1
    assert(count==0)

    
# Test if there are no Nan/Null entries still left in data_catagorical
def test_case6():
    count = 0
    for i in categorical:
        for j in data_categorical[i]:
            if math.isnan(float(j)): 
                count += 1
    assert(count==0)
    
    
# Testing if number of rows and columns of train dataset after concatenation remains correct (input dataset always remains the same, so this # test should always satisfy
def test_case7():
    assert(len(X_train.index) == 25450 and len(X_train.columns) == 1005)
    
    
# Testing if number of rows and columns of test dataset after concatenation remains correct (input dataset always remains the same, so this # test should always satisfy
def test_case8():
    assert(len(X_test.index) == 10908 and len(X_test.columns) == 1005)
    


