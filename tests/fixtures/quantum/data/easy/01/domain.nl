I can carry out the following actions:
map initial a lqubit A to a pqubit B
swap a lqubit A with a lqubit B between a pqubit C and a pqubit D at a depth E
swap ancillary a lqubit A from a pqubit B to a pqubit C at a depth D
apply cnot between a lqubit A mapped to a pqubit C and a lqubit B mapped to a pqubit D at a depth E
move from a depth A to a depth B

I have the following restrictions on my actions:
I can only map initial a lqubit A to a pqubit B if it is not the case that A is an occupied logical qubit and it is not the case that B is initialized
I can only swap a lqubit A with a lqubit B between a pqubit C and a pqubit D at a depth E if it is the case that A is mapped to C and B is mapped to D and the current depth is E and C is connected to D
I can only swap a lqubit A with a lqubit B between a pqubit C and a pqubit D at a depth E if it is not the case that the current depth is d0
I can only swap ancillary a lqubit A from a pqubit B to a pqubit C at a depth D if it is the case that A is mapped to B and the current depth is D and B is connected to C
I can only swap ancillary a lqubit A from a pqubit B to a pqubit C at a depth D if it is not the case that C is an occupied pqubit and it is not the case that the current depth is d0
I can only apply cnot between a lqubit A mapped to a pqubit C and a lqubit B mapped to a pqubit D at a depth E if it is the case that A is mapped to C and B is mapped to D and C is connected to D and A is not reachable from B in direction E and the current depth is E
I can only move from a depth A to a depth B if it is the case that the current depth is A and the next depth after A is B
I can only move from a depth A to a depth B if it is not the case that the current depth is B

The actions have the following effects on the state:
Once I map initial a lqubit A to a pqubit B, it is the case that A is an occupied logical qubit and B is an occupied pqubit and B is initialized and A is mapped to B
Once I swap a lqubit A with a lqubit B between a pqubit C and a pqubit D at a depth E, it is the case that A is mapped to D and B is mapped to C
Once I swap a lqubit A with a lqubit B between a pqubit C and a pqubit D at a depth E, it is not the case anymore that A is mapped to C and it is not the case anymore that B is mapped to D
Once I swap ancillary a lqubit A from a pqubit B to a pqubit C at a depth D, it is the case that A is mapped to C and C is an occupied pqubit and C is initialized
Once I swap ancillary a lqubit A from a pqubit B to a pqubit C at a depth D, it is not the case anymore that A is mapped to B and it is not the case anymore that B is an occupied pqubit
Once I apply cnot between a lqubit A mapped to a pqubit C and a lqubit B mapped to a pqubit D at a depth E, it is not the case anymore that A is not reachable from B in direction E
Once I move from a depth A to a depth B, it is the case that the current depth is B
Once I move from a depth A to a depth B, it is not the case anymore that the current depth is A

Everything that is a pqubit or a lqubit or a depth is also a object
