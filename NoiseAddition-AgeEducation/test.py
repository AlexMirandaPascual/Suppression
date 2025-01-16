from principal_function import *


for eps in [0.25, 0.5, 0.75, 1, 2]:
	for m in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
		for M in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
			if m<=M:
				try:
				    calculate_eps_suppression_inverse(m, M, eps)
				except RuntimeWarning:
				    print("m=",m," M=",M)
				