import pytest

from mypkg.price_range_prediction import price_range

# Testing if feature encoding for price range is correct
def test_case9():
    res = 3
    assert(res == price_range(300583.903))
    
def test_case10():
    res = 1
    assert(res == price_range(100.0))
    
def test_case11():
    res = 7
    assert(res == price_range(5000000))
    
def test_case12():
    res = 4
    assert(res == price_range(605000.77))
    
def test_case13():
    res = 6
    assert(res == price_range(1240000))
    
def test_case14():
    res = 2
    assert(res == price_range(222222))
    
def test_case15():
    res = 5
    assert(res == price_range(800999.55))

