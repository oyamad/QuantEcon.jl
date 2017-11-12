#=
Type definitions

=#

abstract type DPAlgorithm end

"""
This refers to the Value Iteration solution algorithm.

References
----------

https://lectures.quantecon.org/py/discrete_dp.html

"""
struct VFI <: DPAlgorithm end

"""
This refers to the Policy Iteration solution algorithm.

References
----------

https://lectures.quantecon.org/py/discrete_dp.html

"""
struct PFI <: DPAlgorithm end

"""
This refers to the Modified Policy Iteration solution algorithm.

References
----------

https://lectures.quantecon.org/py/discrete_dp.html

"""
struct MPFI <: DPAlgorithm end
