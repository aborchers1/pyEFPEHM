import numpy as np
cimport numpy as np
cimport cython

cpdef my_cgroup_idxs_by_vals(np.ndarray[np.int64_t, ndim=1] arr):
	'''
	Group the positions of `arr` by their value using a counting sort.

	`arr` is a 1D array of non-negative int64 values. Returns a list with one
	array per value v = 0, 1, ..., arr.max(), each holding the positions i with
	arr[i] == v in increasing order of i. Values absent from `arr` give empty
	groups. Runs in O(len(arr) + arr.max()) time.
	'''

	#count how many times each value appears in arr (length arr.max()+1)
	cdef np.ndarray[np.int64_t, ndim=1] a_counts = np.bincount(arr)

	#grouped_idxs receives the positions ordered by value; a_idxs starts as the
	#write cursor at the start offset of each value's group
	cdef np.ndarray[np.int64_t, ndim=1] grouped_idxs = np.empty_like(arr, dtype=np.int64)
	cdef np.ndarray[np.int64_t, ndim=1] a_idxs = np.zeros_like(a_counts, dtype=np.int64)
	a_idxs[1:] = np.cumsum(a_counts[:-1])

	#C-level variables and typed memoryviews for a pure-C scatter loop
	cdef Py_ssize_t i
	cdef np.int64_t a
	cdef np.int64_t[:] grouped_idxs_view = grouped_idxs
	cdef np.int64_t[:] a_idxs_view = a_idxs
	cdef np.int64_t[:] arr_view = arr

	#place each position i in its value's slot and advance that slot's cursor
	for i in range(arr.shape[0]):
		a = arr_view[i]
		grouped_idxs_view[a_idxs_view[a]] = i
		a_idxs_view[a] += 1

	#a_idxs now holds each group's end offset. Use np.split to return one group per value (last entry is len(arr) and would give an empty group)
	return np.split(grouped_idxs, a_idxs[:-1])

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
