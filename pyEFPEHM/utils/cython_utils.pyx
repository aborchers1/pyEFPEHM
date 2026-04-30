import numpy as np
cimport numpy as np
cimport cython

cpdef my_cgroup_idxs_by_vals(np.ndarray[np.int64_t, ndim=1] arr):
    
	#compute how many each index appears in arr
	cdef np.ndarray[np.int64_t, ndim=1] a_counts = np.bincount(arr)
    
	#now generate arrays to store indices and where each index will be located
	cdef np.ndarray[np.int64_t, ndim=1] grouped_idxs = np.empty_like(arr, dtype=np.int64)
	cdef np.ndarray[np.int64_t, ndim=1] a_idxs = np.zeros_like(a_counts, dtype=np.int64)
	a_idxs[1:] = np.cumsum(a_counts[:-1])

	#declare C-level variables for the loop
	cdef Py_ssize_t i
	cdef np.int64_t a

	#create typed memoryviews for direct, C-level buffer access
	cdef np.int64_t[:] grouped_idxs_view = grouped_idxs
	cdef np.int64_t[:] a_idxs_view = a_idxs
	cdef np.int64_t[:] arr_view = arr

	#this loop will now be converted to pure, fast C code
	for i in range(arr.shape[0]):
		a = arr_view[i]
		grouped_idxs_view[a_idxs_view[a]] = i
		a_idxs_view[a] += 1

	return grouped_idxs, a_idxs

#my lightweight cython class to evaluate polynomials using Horners method
cdef class my_cpoly:
	
	#Cython memory view for better performance
	cdef double[:] coefficients
	cdef int N
	
	#initialize polynomial class
	def __cinit__(self, double[:] coeffs):
		#Copy coefficients into internal array, fliping them for Horner method
		self.coefficients = coeffs[::-1].copy()
		self.N = len(self.coefficients)
		
	def __call__(self, double x) -> double:
		#initialize the result
		cdef double result = self.coefficients[0]
		cdef int i
			
		# Use Horner's method to evaluate the polynomial
		for i in range(1, self.N):
			result = result*x + self.coefficients[i]
			
		return result
