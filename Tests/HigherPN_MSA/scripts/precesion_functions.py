import numpy as np
from scipy.integrate import solve_ivp
import scipy.special

try:    import pyEFPEHM
except: pass

t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#function to compute precession frequency of spin 1
def compute_Omega_1(s_2, lN, y, nu, dmu, chi_eff, dchi, pn_spin_order=6):
	
	#compute necessary things
	y2 = y*y
	y4 = y2*y2
	nu2 = nu*nu
	dmuchi1 = dmu*(chi_eff + dchi)
	
	#compute coefficients that enter expressions
	s2_coef         = 0
	const_lN_coef   = 0
	chi_eff_lN_coef = 0
	dchi_lN_coef    = 0
	dmuchi1_lN_coef = 0
	
	if pn_spin_order>=3:
		const_lN_coef   += 7/4 + dmu/4
	if pn_spin_order>=4:
		s2_coef         += 1/2
		chi_eff_lN_coef += -3/2
	if pn_spin_order>=5:
		const_lN_coef   += y2*(33/16 - (61/48)*nu + dmu*(15/16 - nu/48))
	if pn_spin_order>=6:
		s2_coef         += y2*(-nu/4)
		chi_eff_lN_coef += y2*(-5/12 + (3/4)*nu)
		dchi_lN_coef    += 1/12 + (2/3)*nu
		dmuchi1_lN_coef += -31/24
	if pn_spin_order>=7:
		const_lN_coef   += y4*(135/32 - (367/32)*nu + (29/96)*nu2 + dmu*(81/32 - (55/32)*nu - nu2/96))
	if pn_spin_order>=8:
		s2_coef         += y4*(3/8 + (49/16)*nu + nu2/48)
		chi_eff_lN_coef += y4*(-103/32 + (937/144)*nu - nu2/16)
		dchi_lN_coef    += y2*(-31/32 + (65/18)*nu - (7/9)*nu2)
		dmuchi1_lN_coef += y2*(-47/16 + (319/144)*nu)
 		
	return (const_lN_coef + y*(chi_eff*chi_eff_lN_coef + y2*(dchi*dchi_lN_coef + dmuchi1*dmuchi1_lN_coef)))*lN + y*s2_coef*s_2

#function to compute precession frequency of spin 2
def compute_Omega_2(s_1, lN, y, nu, dmu, chi_eff, dchi, pn_spin_order=6):
	return compute_Omega_1(s_1, lN, y, nu, -dmu, chi_eff, -dchi, pn_spin_order=pn_spin_order)

#function to compute DlN + (y**6)*(O1xs1 + O2xs2))
def compute_dDlN(s_1, s_2, lN, lNxs1, lNxs2, y, nu, dmu, chi_eff, dchi, pn_spin_order=6):
	
	#compute necessary things
	y2 = y*y
	nu2 = nu*nu
	dmudchi = dmu*dchi
	
	#compute coefficients that enter expressions
	const_lNxs1ps2_coef   = 0
	dmu_lNxs1ms2_coef     = 0
	chi_lNxs1ps2_coef     = 0
	dmudchi_lNxs1ps2_coef = 0
	dmuchi_lNxs1ms2_coef  = 0
	dchi_lNxs1ms2_coef    = 0
	extra_lN_coef         = 0
	extra_s1_s2_coef      = 0
	
	if pn_spin_order>=5:
		const_lNxs1ps2_coef += 51/16 + (53/48)*nu
		dmu_lNxs1ms2_coef   += -3/16 + (5/48)*nu
	if pn_spin_order>=6:
		chi_lNxs1ps2_coef     += -649/96 - (7/8)*nu
		dmudchi_lNxs1ps2_coef += -31/96
		dmuchi_lNxs1ms2_coef  += 23/96
		dchi_lNxs1ms2_coef    += -7/96 + (7/24)*nu
	if pn_spin_order>=7:
		const_lNxs1ps2_coef += y2*(81/16 - 5*nu - (49/48)*nu2)
		dmu_lNxs1ms2_coef   += y2*(27/16 + (13/16)*nu - nu2/48)
	if pn_spin_order>=8:
		extra_lN_coef         += nu*y2*((28/5)*chi_eff + (4/5)*dmudchi)
		extra_s1_s2_coef       = nu*y2*(-(28/5)*(s_1+s_2) - (4/5)*dmu*(s_1-s_2))
		chi_lNxs1ps2_coef     += y2*(-157/128 + (14339/1152)*nu + (55/96)*nu2)
		dmudchi_lNxs1ps2_coef += y2*(-571/128 - (181/1152)*nu)
		dmuchi_lNxs1ms2_coef  += y2*(-553/128 - (811/1152)*nu)
		dchi_lNxs1ms2_coef    += y2*(-7/128 + (2525/1152)*nu + (7/288)*nu2)
		
	return y*(extra_lN_coef*lN + extra_s1_s2_coef) + (const_lNxs1ps2_coef + y*(chi_eff*chi_lNxs1ps2_coef + dmudchi*dmudchi_lNxs1ps2_coef))*(lNxs1 + lNxs2) + (dmu*dmu_lNxs1ms2_coef + y*(dmu*chi_eff*dmuchi_lNxs1ms2_coef + dchi*dchi_lNxs1ms2_coef))*(lNxs1 - lNxs2)


#function to compute the derivative of angulat momenta
def compute_Ds1_Ds2_DlN(s_1, s_2, lN, y, nu, dmu, pn_spin_order=6):
	
	#make sure s_1, s_2 and lN are numpy arrays
	s_1 = np.asarray(s_1)
	s_2 = np.asarray(s_2)
	lN  = np.asarray(lN)
	
	#compute necessary things
	chi_1 = np.dot(s_1, lN)
	chi_2 = np.dot(s_2, lN)
	chi_eff = chi_1 + chi_2
	dchi    = chi_1 - chi_2
	y2 = y*y
	y5 = y2*y2*y
	lNxs1 = np.cross(lN , s_1)
	lNxs2 = np.cross(lN , s_2)
	s1xs2 = np.cross(s_1, s_2)
	
	#compute precession frequencies
	O1xs1 = compute_Omega_1(-s1xs2, lNxs1, y, nu, dmu, chi_eff, dchi, pn_spin_order=pn_spin_order)
	O2xs2 = compute_Omega_2( s1xs2, lNxs2, y, nu, dmu, chi_eff, dchi, pn_spin_order=pn_spin_order)
	
	#compute evolution equations for spins
	Ds1 = 0.5*(1 - dmu)*y5*O1xs1
	Ds2 = 0.5*(1 + dmu)*y5*O2xs2

	#compute evolution equation for direction of Newtonian angular momentum
	dDlN = compute_dDlN(s_1, s_2, lN, lNxs1, lNxs2, y, nu, dmu, chi_eff, dchi, pn_spin_order=pn_spin_order)
	DlN = y*y5*(y2*dDlN - (O1xs1 + O2xs2))
	
	#return the derivatives
	return Ds1, Ds2, DlN

#function to compute orbital angular momentum
def compute_L(s_1, s_2, lN, y, nu, dmu, pn_spin_order=6):

	#compute necessary things
	y2 = y*y
	y3 = y2*y
	y4 = y2*y2
	nu2 = nu*nu
	s1ps2 = s_1 + s_2
	s1ms2 = s_1 - s_2
	lNs1 = np.sum(lN*s_1, axis=0)
	lNs2 = np.sum(lN*s_2, axis=0)
	s2_1 = np.sum(s_1*s_1, axis=0)
	s2_2 = np.sum(s_2*s_2, axis=0)
	s1s2 = np.sum(s_1*s_2, axis=0)
	chi_eff = lNs1 + lNs2
	dchi    = lNs1 - lNs2
	s21ps22 = s2_1 + s2_2
	s21ms22 = s2_1 - s2_2
	dmudchi = dmu*dchi
	chi_eff2 = chi_eff*chi_eff
	
	#compute coefficients that enter expressions
	lN_coef    = 0
	s1ps2_coef = 0
	s1ms2_coef = 0
	
	if pn_spin_order>=5:
		lN_coef    += 3/2 + nu/6
		s1ps2_coef += 7/4
		s1ms2_coef += 1/4
	if pn_spin_order>=6:
		lN_coef    += y*((-49/24)*chi_eff + (-7/24)*dmudchi - y*s1s2)
		s1ps2_coef += y*(-chi_eff)
	if pn_spin_order>=7:
		lN_coef    += y2*(27/8 - (19/8)*nu + nu2/24 - s21ps22/2 + chi_eff2)
		s1ps2_coef += y2*(33/16 - nu*(61/48))
		s1ms2_coef += y2*(15/16 - nu*(1/48))
	if pn_spin_order>=8:
		lN_coef    += y3*((-121/32 + nu*(671/288))*chi_eff + (-55/32 + nu*(11/288))*dmudchi + y*nu*(2/3)*s1s2)
		s1ps2_coef += y4*((-119/48 + nu/2)*chi_eff + (7/48)*dmudchi)
		s1ms2_coef += y4*((7/48)*chi_eff + (1/48)*dmudchi)
		

	return (nu/y)*(1 + y2*lN_coef)*lN - 0.5*nu*y2*(s1ps2_coef*s1ps2 + s1ms2_coef*dmu*s1ms2)

#function to compute total angular momentum
def compute_J(s_1, s_2, lN, y, nu, dmu, pn_spin_order=6):
	
	#compute L, S_1 and S_2
	L = compute_L(s_1, s_2, lN, y, nu, dmu, pn_spin_order=pn_spin_order)
	S_1 = 0.5*(1 + dmu)*s_1
	S_2 = 0.5*(1 - dmu)*s_2
	
	#return the total angular moment
	return L + S_1 + S_2

#function to evolve spins
def evolve_precessing_system_noRR(s_1_0, s_2_0, lN_0, y, nu, dmu, tmax, pn_spin_order=6, rtol=1e-10, atol=1e-10, method='DOP853'):
	
	#function to compute derivatives of v = [s_1, s_2, lN]
	def dv_dt(t, v):
		
		#compute the derivatives of the angular momenta
		Ds1, Ds2, DlN = compute_Ds1_Ds2_DlN(v[:3], v[3:6], v[6:9], y, nu, dmu, pn_spin_order=pn_spin_order)
		
		#return dv/dy
		return np.array([*Ds1, *Ds2, *DlN])

	#return solution to system of differential equations
	return solve_ivp(dv_dt, [0, tmax], [*s_1_0, *s_2_0, *lN_0], dense_output=True, method=method, rtol=rtol, atol=atol)

#function to initialize pyEFPE PN derivatives
def initialize_pyEFPE_PN_derivatives(s_1, s_2, lN, y, nu, dmu, pn_phase_order=6, pn_spin_order=6):

	#make sure s_1, s_2 and lN are numpy arrays
	s_1 = np.asarray(s_1)
	s_2 = np.asarray(s_2)
	lN  = np.asarray(lN)
	
	#compute necessary things
	chi_1 = np.dot(s_1, lN)
	chi_2 = np.dot(s_2, lN)
	chi_eff = chi_1 + chi_2
	dchi    = chi_1 - chi_2
	s2_1 = np.sum(np.square(s_1), axis=0)
	s2_2 = np.sum(np.square(s_2), axis=0)
	
	#return initialized pyEFPE class to compute derivatives
	return pyEFPEHM.functions.pyEFPE_PN_derivatives(0.5*(1 + dmu), 0.5*(1 - dmu), s2_1, s2_2, pn_phase_order=pn_phase_order, pn_spin_order=pn_spin_order)

#function to compute PN derivatives
def compute_Dy_De2_Dlambda_Ddlambda(PN_derivatives, s_1, s_2, lN, y, e2):

	#make sure s_1, s_2 and lN are numpy arrays
	s_1 = np.asarray(s_1)
	s_2 = np.asarray(s_2)
	lN  = np.asarray(lN)
	
	#compute necessary things
	chi_1 = np.dot(s_1, lN)
	chi_2 = np.dot(s_2, lN)
	chi_eff = chi_1 + chi_2
	dchi    = chi_1 - chi_2
	s2 = np.sum(np.square(s_1 + s_2), axis=0)

	#return derivatives
	return PN_derivatives.Dy_De2_Dl_Ddl(y, e2, chi_eff, dchi, dchi*dchi, s2 - chi_eff*chi_eff)

#function to compute evolution of Euler angles between lN and J
def compute_Dphiz_Dzeta_lNJ(s_1, s_2, lN, DlN, y, nu, dmu, pn_spin_order=6):

	#make sure s_1, s_2 and lN are numpy arrays
	s_1 = np.asarray(s_1)
	s_2 = np.asarray(s_2)
	lN  = np.asarray(lN)

	#compute total angular momentum
	J = compute_J(s_1, s_2, lN, y, nu, dmu, pn_spin_order=pn_spin_order)
	J_norm = np.linalg.norm(J, axis=0)
	j_vec = J/J_norm

	#compute cosine of the angle between lN and j
	cth = np.dot(lN, j_vec)
	sth2 = 1 - cth*cth
	
	#compute the derivative of phiz
	Dphiz = np.divide(np.dot(DlN, np.cross(j_vec , lN)), sth2, where=(sth2 != 0), out=np.zeros_like(sth2))
	#compute the derivative of zeta from minimal rotation condition
	Dzeta = -cth*Dphiz
	
	return Dphiz, Dzeta

#function to evolve spins
def evolve_precessing_system_with_RR(y0, yf, e20, s_1_0, s_2_0, lN_0, nu, dmu, lamb0=0, dlamb0=0, phiz0=0, zeta0=0, pn_spin_order=6, pn_phase_order=6, rtol=1e-10, atol=1e-10, method='DOP853'):
	
	#initialize PN derivatives
	PN_derivatives = initialize_pyEFPE_PN_derivatives(s_1_0, s_2_0, lN_0, y0, nu, dmu, pn_phase_order=pn_phase_order, pn_spin_order=pn_spin_order)
	
	#function to compute derivatives of v = [*s_1, *s_2, *lN, e2, lambda, dlambda, phiz, zeta] with respect to y
	def dv_dy(y, v):
		
		#compute the derivatives with respect to time of the angular momenta
		Ds1, Ds2, DlN = compute_Ds1_Ds2_DlN(v[:3], v[3:6], v[6:9], y, nu, dmu, pn_spin_order=pn_spin_order)
		
		#compute derivatives with respect to y
		Dy, De2, Dlamb, Ddlamb = compute_Dy_De2_Dlambda_Ddlambda(PN_derivatives, v[:3], v[3:6], v[6:9], y, v[9])

		#compute also the evolution of the Euler angles
		Dphiz, Dzeta = compute_Dphiz_Dzeta_lNJ(v[:3], v[3:6], v[6:9], DlN, y, nu, dmu, pn_spin_order=pn_spin_order)
		
		#return dv/dy (note that phase evolution is turned off because it was a bit of a pain)
		return np.array([*Ds1, *Ds2, *DlN, De2, 0*Dlamb, 0*Ddlamb, Dphiz, Dzeta])/Dy

	#return solution to system of differential equations
	return solve_ivp(dv_dy, [y0, yf], [*s_1_0, *s_2_0, *lN_0, e20, lamb0, dlamb0, phiz0, zeta0], dense_output=True, method=method, rtol=rtol, atol=atol)

#function to compute the evolution of dynamical quantities from pyEFPE
def compute_quantities_with_pyEFPE(ys, e0, s_1_0, s_2_0, lN_0, nu, dmu, lamb0=0, dlamb0=0, phiz0=0, zeta0=0, pn_spin_order=6, pn_phase_order=6):

	#make sure initial s_1, s_2 and lN are numpy arrays
	s_1_0 = np.asarray(s_1_0)
	s_2_0 = np.asarray(s_2_0)
	lN_0  = np.asarray(lN_0)
	
	#for now, make sure that initial orb. ang. mom. is in z-direction
	assert np.all(lN_0 == np.array([0,0,1]))

	#compute mu1 and mu2
	mu1, mu2 = 0.5*(1+dmu), 0.5*(1-dmu)

	#compute dimensionless spins
	chi1_0, chi2_0 = s_1_0/mu1, s_2_0/mu2

	#initialize pyEFPE parameters
	params = {
	'mass1': mu1/t_sun_s,
	'mass2': mu2/t_sun_s,
        'e_start': e0,
        'spin1x': chi1_0[0],
        'spin1y': chi1_0[1],
        'spin1z': chi1_0[2],
        'spin2x': chi2_0[0],
        'spin2y': chi2_0[1],
        'spin2z': chi2_0[2],
        'f22_start': ((((1-e0*e0)**0.5)*np.amin(ys))**3)/np.pi,
        'f22_end': (np.amax(ys)**3)/np.pi,
        'Interpolate_Amplitudes': False,
        'pn_phase_order': pn_phase_order,
        'pn_spin_order': pn_spin_order,
	}

	#initialize waveform
	wf = pyEFPEHM.pyEFPE(params)
	
	#compute array of times
	times = -np.flip(np.geomspace(-wf.sol.all_ts[-1], -wf.sol.all_ts[0], len(ys)))

	#compute the dynamical variables
	y, e2, DJ2, bpsip, phiz, zeta = wf.sol(times, idxs=[0,1,4,5,6,7])

	#compute the Euler angles
	wf.MSA.update(y, DJ2)
	dphiz, dzeta, costhL = wf.MSA.precession_Euler_angles(bpsip)
	phiz += dphiz
	zeta += dzeta
	
	#compute dchi copying what is used inside precession_Euler_angles
	hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
	hpsip = wf.MSA.K_m*hbpsip_pi_2
	sn, cn, dn, am = scipy.special.ellipj(hpsip, wf.MSA.m)
	dchi = wf.MSA.dchi_av - wf.MSA.dchi_diff*(1 - 2*sn*sn)
	
	#compute chi_eff with pyEFPE
	chi_eff = wf.MSA.chi_eff0 + wf.MSA.kchi*(dchi - wf.MSA.dchi0)
	
	#add variations of DJ2 from precession average
	DJ2 += wf.MSA.compute_dDJ2(hpsip, e2)
	
	#return the full Euler angles
	return wf, np.interp(ys, y, e2), np.interp(ys, y, phiz), np.interp(ys, y, zeta), np.interp(ys, y, costhL), np.interp(ys, y, dchi), np.interp(ys, y, chi_eff), np.interp(ys, y, DJ2)

#function to compute the exact DJ2
def compute_DJ2_exact_wth_pyEFPE(ys, J2, wf_pyEFPE):
	
	#compute parallel component of total angular momentum J
	J0lN = wf_pyEFPE.MSA.aJ + wf_pyEFPE.MSA.bJ*wf_pyEFPE.MSA.chi_eff0 + wf_pyEFPE.MSA.cJ*wf_pyEFPE.MSA.dchi0
	J0lN2 = J0lN*J0lN

	#compute squared perpedicular component of J
	bJpcJ = wf_pyEFPE.MSA.bJ + wf_pyEFPE.MSA.cJ
	bJpcJ2 = bJpcJ*bJpcJ
	bJmcJ = wf_pyEFPE.MSA.bJ - wf_pyEFPE.MSA.cJ
	bJmcJ2 = bJmcJ*bJmcJ
	Sp2_1 = bJpcJ2*wf_pyEFPE.MSA.sp2_1
	Sp2_2 = bJmcJ2*wf_pyEFPE.MSA.sp2_2

	#below 2.5PN, bJ2_m_cJ2 can be a float, make sure it has the correct shape
	if np.asarray(wf_pyEFPE.MSA.bJ2_m_cJ2).ndim==0:
		bJ2_m_cJ2_pyEFPE = np.full(len(ys), wf_pyEFPE.MSA.bJ2_m_cJ2)
	else:
		bJ2_m_cJ2_pyEFPE = wf_pyEFPE.MSA.bJ2_m_cJ2

	#compute things in the same y-grid as the exact solution
	J2_no_DJ2 = np.interp(ys, wf_pyEFPE.MSA.y, J0lN2 + Sp2_1 + Sp2_2)
	bJ2_m_cJ2 = np.interp(ys, wf_pyEFPE.MSA.y, bJ2_m_cJ2_pyEFPE)

	return (J2 - J2_no_DJ2)/bJ2_m_cJ2

#function to convert from cartesian to spherical coordinates
def cartesian_2_spherical(x, y, z):
	
	#compute the radious
	r = np.sqrt(x*x + y*y + z*z)
	
	#compute polar angle
	th = np.arccos(z/r)
	
	#compute the azimutal angle
	ph = np.arctan2(y, x)
	
	#return the spherical coordinates
	return r, th, ph
