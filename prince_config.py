import sys
import platform
import os.path as path
base = path.dirname(path.abspath(__file__))
sys.path.append(base)
# sys.path.append(base+"/CRFluxModels")


#detrmine shared library extension and MKL path
lib_ext = None
mkl_default = path.join(sys.prefix, 'lib', 'libmkl_rt')

if platform.platform().find('Linux') != -1:
    lib_ext = '.so'
elif platform.platform().find('Darwin') != -1:
    lib_ext = '.dylib'
else:
    #Windows case
    mkl_default = path.join(sys.prefix, 'pkgs', 'mkl-11.3.3-1',
                            'Library','bin', 'mkl_rt')
    lib_ext = '.dll'

config = {

# Debug flag for verbose printing, 0 = minimum
"debug_level": 1,

# Use progress_bars
"prog_bar": False,


#=========================================================================
# Paths and library locations
#=========================================================================

# Directory where the data files for the calculation are stored
"data_dir": base+'/data',

# # File name of particle production yields
# "yield_fname": "yield_dict.ppd",

# full path to libmkl_rt.[so/dylib] (only if kernel=='MKL')
"MKL_path": mkl_default + lib_ext,

#=========================================================================
# Physics configuration
#=========================================================================
# Cosmological parameters

# Hubble constant
"H_0" : 70.5, #km s^-1 Mpc^-1
"H_0s" : 2.28475e-18, #s^-1

# Omega_m
"Omega_m" : 0.27,

# Omega_Lambda
"Omega_Lambda" : 0.73,

#===========================================================================
# Parameters of numerical integration
#===========================================================================

# Selection of integrator (euler/odepack)   
"integrator": "euler",

# euler kernel implementation (numpy/MKL/CUDA).
"kernel_config": "MKL",

#parameters for the odepack integrator. More details at
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode
"ode_params": {'name':'vode',
               'method':'adams',
               'nsteps':10000,
               'max_step':10.0},

# Use sparse linear algebra (recommended!)
"use_sparse": True,

#Number of MKL threads (for sparse matrix multiplication the performance
#advantage from using more than 1 thread is limited by memory bandwidth)
"MKL_threads": 24,

# Float precision (32 only yields speed up with CUDA, MKL gets slower?)
"FP_precision": 64,

#=========================================================================
# Advanced settings
#=========================================================================

# Ratio of decay_length/interaction_length where particle interactions
# are neglected and the resonance approximation is used
"hybrid_crossover": 0.05,

# Muon energy loss according to Kokoulin et al.
"enable_muon_energy_loss": True,

# Minimal stepsize for muon energy loss steps
"muon_energy_loss_min_step": 5.,

# First interaction mode 
# (stop particle production after one interaction length)
"first_interaction_mode": False,

# Possibilities to control the solver (some options are obsolete/not
# working)
"vetos": {
    # inhibit coupling/secondary production of mesons
    "veto_sec_interactions": False,
    # Disable resonance/prompt contribution
    "veto_resonance_decay": False,
    "veto_hadrons": [],
    "veto_resonances": [],
    "allow_resonances": [],
    'veto_forward_mesons':False,
    # Switch off decays. E.g., disable muon decay with [13,-13]
    "veto_decays": [],
    # Switch off particle production by charm projectiles
    "veto_charm_pprod": False, 
    # Disable mixing of resonance approx. and propagation
    "no_mixing": False
    }
}



dbg = config['debug_level']

def mceq_config_without(key_list):
    r = dict(config)  # make a copy
    for key in key_list:
        del r[key]
    return r
