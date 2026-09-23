# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:21:35 2025

@author: jarom
"""

import functools
"""https://www.youtube.com/watch?v=r7Dtus7N4pI&t=1s&ab_channel=Kite"""
def my_decorator(func):
    def wrapper(*args,**kwargs):
        print("Started")
        val = func(*args,**kwargs)
        print("Ended")
        return val
    return wrapper

@my_decorator
def f(a):
    """This function says hello."""
    print(a)
@my_decorator
def add(x,y):
    return x+y




print(add(4,5))




"""https://www.youtube.com/watch?v=4jBJhCaNrWU&ab_channel=b001"""

def order_pizza(size,*toppings, **details):
    print(f"Ordered a {size} pizza with the folowing toppings:")
    for topping in toppings:
        print(f" - topping")
        
    print(f"\nDetails of the order are:")
    for key, value in details.items():
        print(f"- {key}: {value}")
    return toppings, details
        
toppings,details=order_pizza("large", "pepperoni", "olives",delivery=True,  tip=5,time="nove")


