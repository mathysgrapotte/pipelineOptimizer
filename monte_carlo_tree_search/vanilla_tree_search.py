"""
Okay so here we are trying to implement a montecarlo tree search in its vanilla form, so to really understand how that works:

First we consider a start state, in our case that would be no parameter setup with the choice of parameter one two or three. 
Interestingly we could ask the tree to setup parameters in a different order right ? 
Although in terms of backprop, I am unsure on how that would work, that's an idea to keep for another time.




"""

from collections import defaultdict

