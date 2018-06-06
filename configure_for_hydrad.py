"""
Use the loops from our active region to configure a HYDRAD simulation
"""
import argparse
import os
import sys

import astropy.units as u
import synthesizAR

parser = argparse.ArgumentParser()
parser.add_argument('--loop_number', help='Number indicating which loop to select',type=int)
parser.add_argument('--interface_path', help='Path to HYDRAD interface file',type=str)
parser.add_argument('--ar_path', help='Path to saved active region', type=str)
parser.add_argument('--hydrad_path', help='Path to HYDRAD source', type=str)
parser.add_argument('--results_path', help='Path to configured simulation', type=str)
args = parser.parse_args()

sys.path.append(args.interface_path)
from hydrad_interface import HYDRADInterface

# Load active region
active_region = synthesizAR.Field.restore(args.ar_path)
# Setup base configuration
base_config = {
    'general': {
        'total_time': 5e4 * u.s,
        #'loop_length': 80 * u.Mm,
        'footpoint_height': 5e8 * u.cm,
        'output_interval': 5*u.s,
        'loop_inclination': 0*u.deg,
        'logging_frequency': 1000,
        'write_file_physical': True,
        'write_file_ion_populations': False,
        'write_file_hydrogen_level_populations': False,
        'write_file_timescales': False,
        'write_file_equation_terms': False,
        'heat_flux_limiting_coefficient': 1./6.,
        'heat_flux_timestep_limit': 1e-10*u.s,
        'use_kinetic_model': False,
        'minimum_collisional_coupling_timescale': 0.01*u.s,
        'force_single_fluid': False,
        #'tabulated_gravity_profile':,
        #'tabulated_cross_section_profile':,
    },
    'initial_conditions': {
        'footpoint_temperature': 2e4 * u.K,
        'footpoint_density': 1e11 * u.cm**(-3),
        'isothermal': False,
        #'heating_location':,
        'heating_scale_height': 1e300*u.cm,
        'heating_range_lower_bound': 1e-8*u.erg/u.s/(u.cm**3),
        'heating_range_upper_bound': 1e2*u.erg/u.s/(u.cm**3),
        'heating_range_step_size': 0.01,
        'heating_range_fine_tuning': 10000.0,
        'use_tabulated_gravity': False,
    },
    'radiation': {
        'use_power_law_radiative_losses': True,
        'decouple_ionization_state_solver': False,
        'density_dependent_rates': False,
        'optically_thick_radiation': False,
        'nlte_chromosphere': False,
        'ranges_dataset': 'ranges',
        'emissivity_dataset': 'chianti_v7',
        'abundance_dataset': 'asplund',
        'rates_dataset': 'chianti_v7',
        'elements_equilibrium': ['Fe'],
        'elements_nonequilibrium': [],
    },
    'heating': {
        'heat_electrons': True,
        'background_heating': True,
        'beam_heating': False,
        'alfven_wave_heating': False,
        #'events': [
        #    {'time_start': 0.*u.s, 
        #     'rise_duration': 100*u.s,
        #     'decay_duration': 100*u.s, 
        #     'total_duration': 200*u.s,
        #     'location': 4e9*u.cm, 
        #     'scale_height': 1e300 * u.cm,
        #     'rate': 0.1 *u.erg/u.s/(u.cm**3),},
        #],
    },
    'solver': {
        'epsilon': 0.01,
        'safety_radiation': 0.1,
        'safety_conduction': 1.0,
        'safety_advection': 1.0,
        'safety_atomic': 1.0,
        'safety_viscosity': 1.0,
        'cutoff_ion_fraction':1e-6,
        'epsilon_d':0.1,
        'epsilon_r':1.8649415311920072,
        'timestep_increase_limit': 5*u.percent,
        'relative_viscous_timescale': None,
        'minimum_radiation_temperature': 2e4*u.K,
        'zero_over_temperature_interval': 5.0e2*u.K,
        'minimum_temperature': 1e4*u.K,
        'maximum_optically_thin_density': 1e12*u.cm**(-3),
    },
    'grid': {
        'adapt': True,
        'adapt_every_n_time_steps': 10,
        #'minimum_cells': 150,
        'maximum_cells': 30000,
        'maximum_refinement_level': 12,
        'minimum_delta_s': 1.0*u.cm,
        'maximum_variation': 10*u.percent,
        'refine_on_density': True,
        'refine_on_electron_energy': True,
        'refine_on_hydrogen_energy': False,
        'minimum_fractional_difference': 10*u.percent,
        'maximum_fractional_difference': 20*u.percent,
        'linear_restriction': True,
        'enforce_conservation': False,
    }
} 
# Create interface
interface = HYDRADInterface(base_config, args.hydrad_path, args.results_path)
# Configure simulation
interface.configure_input(active_region.loops[args.loop_number])
