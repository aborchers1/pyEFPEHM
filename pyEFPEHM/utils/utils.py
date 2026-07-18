import numpy as np
import scipy.special
import warnings

from pyEFPEHM.utils.constants import *

# Function to compute spherical coordinates from vector
def spherical_coords_from_vec(vec, return_trigonometric=True):
	#treat vec as numpy array
	vec = np.asarray(vec)
	
	#make sure vec has the correct shape
	assert vec.shape == (3,)
	
	#make sure the vector is unitary
	vec = vec/np.linalg.norm(vec)	

	#compute the cosine of theta from z component
	cth = vec[2]
	
	#compute the phi angle from y and x components
	ph = np.arctan2(vec[1],vec[0])
	
	#choose to return sine and cosine of theta and phi
	if return_trigonometric:
		return (1 - cth*cth)**0.5, cth, np.sin(ph), np.cos(ph)
	#otherwise, return cos(theta) and phi	
	else:
		return cth, ph


#function to compute {x^(i+1), i=0,..,N} efficiently
def power_range(x, nmax):

	#force x to be a numpy array
	x = np.asarray(x)
	
	#differentiate cases of x having 0 or 1 dimensions
	if   x.ndim==0: return np.cumprod(np.tile(x, nmax))
	elif x.ndim==1: return np.cumprod(np.tile(x,(nmax, 1)), axis=0)
	else: raise Exception('x has %s dimensions. Maximum expected number of dimensions: 1'%(x.ndim))

#my lightweight class to evaluate polynomials using Horners method
class my_poly:
	
	#initialize with coefficient array p(x) = sum_i coefs[i] x^i
	def __init__(self, coefs):
		#flip the coefficients such that p(x) = sum_i coefs[i] x^(n - i)
		#we convert coefficients to a tuple for speed reasons
		self.coefs = tuple(np.flip(coefs))
	
	#method to evaluate the polynomial efficiently using Horners method
	def __call__(self, x):
		#initialize the result to the (x^n) coefficient
		result = self.coefs[0]
		#Apply Horners method
		for coef in self.coefs[1:]:
			result = coef + result*x
		#return the result
		return result

#do my custom version of numpy where which is more optimized than numpy where when codition is a boolean
#Where condition is True, yield true_value, otherwise yield false_value
def my_where(condition, true_value, false_value):
	
	#if the condition is a bool, do a python optimized where
	if isinstance(condition, (bool, np.bool_)):
		if condition: return true_value
		else:         return false_value
	
	#otherwise use numpy.where
	else:
		return np.where(condition, true_value, false_value)

#function to perform safe division by scalar type objects
def safe_divide_by_scalar(num, den, min_den=0., out=0.):
	if abs(den) > min_den: return num/den
	else                 : return out

#function to perform safe division by array type objects
def safe_divide_by_array(num, den, min_den=0., out=None, default_dtype=np.float64):
	#if required, make out have the correct shape and value
	if out is None:
		out = np.zeros(np.broadcast(num, den).shape, dtype=default_dtype)
	elif np.isscalar(out):
		out = np.full( np.broadcast(num, den).shape, out, dtype=default_dtype)
	#perform safe division
	return np.divide(num, den, out=out, where=np.abs(den)>min_den)

#create my own class structure to save the dense result of scipy.integrate.solve_ivp
class ivp_sol_interp:
	
	#method to initialize class from a scipy.integrate.OdeSolution object
	#interpolant times are assumed to be ordered!
	def __init__(self, sol, t_final=None):
			
		#save all points (including last one)
		self.all_ts = sol.ts_sorted
		
		#store also the attributes of the interpolants in lists
		self.ts = list()
		self.hs = list()
		#for ys and Qs consider 0th, 1st and 2nd derivatives
		self.ys = [[], [], []]
		self.Qs = [[], [], []]
		#loop over interpolants
		for interpolant in sol.interpolants:
			#store value of t_n
			self.ts.append(interpolant.t_old)
			#store the value of y at previous interpolation node
			self.ys[0].append(interpolant.y_old)
			#extract value of step h_n = t_{n+1} - t_n
			h = interpolant.h
			#store step value
			self.hs.append(h)
			#extract values of polynomial coefficients
			Q_temp = interpolant.Q
			#store value of polynomial coefficients
			self.Qs[0].append(h*Q_temp)
			#Compute polynomial coeficients for 1st derivative
			iQ = np.arange(Q_temp.shape[1])[np.newaxis,:]
			Q_temp = (1+iQ)*Q_temp
			self.Qs[1].append(Q_temp[:,1:])
			self.ys[1].append(Q_temp[:,0])
			#Compute polynomial coeficients for 2nd derivative
			iQ = np.arange(Q_temp.shape[1]-1)[np.newaxis,:]
			Q_temp = (1+iQ)*Q_temp[:,1:]/h
			self.Qs[2].append(Q_temp[:,1:])
			self.ys[2].append(Q_temp[:,0])

		#convert to numpy arrays
		self.ts = np.array(self.ts)
		self.hs = np.array(self.hs)
		for derivative in [0,1,2]:
			self.ys[derivative] = np.array(self.ys[derivative])
			self.Qs[derivative] = np.array(self.Qs[derivative])
		
		#check that times are indeed sorted
		if not np.all(self.ts[:-1]<=self.ts[1:]):
			raise Exception("Interpolants are not sorted in time")
		
		#if required, shift the time such that the last time corresponds to t_final
		if t_final is not None:
			
			#compute the amount we have to shift the time by
			t_shift = t_final - self.all_ts[-1]

			#update the times
			self.all_ts += t_shift
			self.ts += t_shift
		
	#method to evaluate the interpolant or any derivative of it
	def __call__(self, t, derivative=0, idxs=None):

		#if indexes is None, select all indxs
		if idxs is None: idxs = np.arange(self.Qs[0].shape[1])
		
		#extract the relevant Q and y
		Q = self.Qs[derivative][:,idxs,:]
		y0 = self.ys[derivative][:,idxs]

		#force t to be a numpy array
		t = np.asarray(t)

		#compute the segments each t is in
		segments = np.searchsorted(self.ts, t, side="right") - 1
		#make sure segments are always greater than 0
		segments = np.maximum(segments, 0)
		
		#compute the x of the interpolants
		x = (t - self.ts[segments])/self.hs[segments]

		#Make sure that x has the correct shape
		if (x.ndim!=0) and (np.asarray(idxs).ndim!=0): x = x[:,None]

		#now compute the result
		result = x*Q[segments,...,-1]
		for iQ in reversed(range(Q.shape[-1]-1)):
			result = x*(Q[segments,...,iQ] + result)
		result += y0[segments]
		
		return np.transpose(result)

#function to compute all mass parameters from m1, m2
def mass_params_from_m1_m2(m1, m2):

	#check that m1>m2
	if m1<m2: warnings.warn("m1 < m2; expected m1 >= m2.", UserWarning, stacklevel=2)

	#now compute mass related stuff
	M = m1 + m2           #total mass
	mu1, mu2 = m1/M, m2/M #reduced individual masses
	nu = mu1*mu2          #symmetric mass ratio 
	dmu = mu1 - mu2       #dimensionless mass diference

	return M, mu1, mu2, nu, dmu	

#function to compute the elliptic integral of the third kind \Pi(n;\phi;m) using Carlsons symmetric form RJ and the elliptic integral of the first kind K(m)
def my_ellipPI(n, m, phi=None):
	
	#if phi is not given, return complete integral
	if phi is None:
		return scipy.special.ellipk(m) + (n/3)*scipy.special.elliprj(0, 1-m, 1, 1-n)
	#otherwise, return the corresponding incomplete integral
	else:
		sphi = np.sin(phi)
		sphi2 = sphi*sphi
		nsphi2 = n*sphi2
		return scipy.special.ellipkinc(phi, m) + (nsphi2/3)*sphi*scipy.special.elliprj(1 - sphi2, 1 - m*sphi2, 1, 1-nsphi2)

#function to find for sorted array x, which values of j are in x0[i] <= x[j] <= xf[i]. This function is equivalent (but faster than)
#i_idxs, x_idxs = np.where((x[np.newaxis,:]>=x0[:,np.newaxis]) & (x[np.newaxis,:]<=xf[:,np.newaxis]))
def sorted_vals_in_intervals(x, x0, xf):
	
	#check that indeed the x are sorted
	if np.any(x[:-1]>x[1:]): raise Exception("Input array of values has to be sorted")

	#find initial and final index in each interval
	i_0 = np.searchsorted(x, x0, side='left')
	i_f = np.searchsorted(x, xf, side='right')
	
	#find intervals where there are values
	valid_idxs = np.nonzero(i_f > i_0)[0]
	
	#lengths of those intervals
	lens = i_f[valid_idxs] - i_0[valid_idxs]
	
	#construct an array with the indexes of the intervals
	i_idxs = np.repeat(valid_idxs, lens)
	
	#construct an array with the index of x each i_idxs corresponds to
	if len(valid_idxs)>0:
		x_idxs = np.concatenate([np.arange(i_0[idx], i_f[idx]) for idx in valid_idxs])
	else:
		x_idxs = np.array([], dtype=int)
	
	return x_idxs, i_idxs

#function to perform series reversion, i.e. given y = a1*x + a2*x^2 + a3*x^3 ..., obtain x = A_1*y + A_2*y**2 + A_3*y**3
#We assume a to be a numpy array containing coefficients [a1, a2, a3, ...]
#If a has shape (n,m), the coefficients are along the second axis
def series_reversion(a, order=5):
	
	#check if the input order is larger than the maximum order implemented
	max_order=5
	if order>max_order:
		warnings.warn("Input order=%s, but series reversion is implemented only up to order=%s. Reverting to that order."%(order, max_order), UserWarning, stacklevel=2)
		order=max_order
	#consider the case in which order==1 separately, since it is significantly easier
	elif order==1:
		return 1/a[...,[0]]
	elif order<=0:
		raise Exception('order=%s not valid. order has to be an integer >=1.'%(order))

	#consider different possible shapes of a
	if a.ndim==2:
		#if a has two dimensions, transpose it for simplicity
		a = np.transpose(a)
		#initialize array of inverse coefficients A transposed w.r.t what is going to be returned
		A = np.ones((order, a.shape[1]))
		#make a consistent with order
		if   len(a)<order:
			a = np.append(a, np.zeros((order-len(a), a.shape[1])), axis=0)
		elif len(a)>order:
			a = a[:order]
	elif a.ndim==1:
		#initialize array of inverse coefficients A:
		A = np.ones(order)
		#make a consistent with order
		if   len(a)<order:
			a = np.append(a, np.zeros(order-len(a)))
		elif len(a)>order:
			a = a[:order]
	else:
		raise Exception('Shape %s of a is invalid. It has to have 1 or 2 dimensions.'%(a.shape))
	
	#separate a1 from the rest and normalize
	a1 = a[0]
	a = a[1:]/a[0,...]

	#now loop over orders
	for i in range(1,order):
		if   i==1:
			A[1] = -a[0]
		elif i==2:
			a02 = a[0]*a[0]
			A[2] = 2*a02 - a[1]
		elif i==3:
			A[3] = 5*a[0]*a[1] - a[2] - 5*a02*a[0]
		elif i==4:
			A[4] = 6*a[0]*a[2] + 3*a[1]*a[1] + 14*a02*a02 - a[3] - 21*a02*a[1]
	
	#compute a1**-(k+1)
	amk = power_range(1/a1, order)
	
	#now compute A[k] = (a1**-(k+1))*A[k]
	if A.ndim==2:
		return np.transpose(amk*A)
	else:
		return amk*A
