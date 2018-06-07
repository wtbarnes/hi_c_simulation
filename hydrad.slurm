#!/usr/bin/bash

### SLURM Parameters
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=hi-c_hydrad
#SBATCH --time=00:10:00
#SBATCH --mail-user=wtb2@rice.edu
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

### Setup needed parameters
printf -v LOOP_NUM "%06d" $SLURM_ARRAY_TASK_ID
RESULTS_DIR=$SHARED_SCRATCH/wtb2/hi_c_simulation
HYDRAD_DIR=$SHARED_SCRATCH/wtb2/hi_c_simulation/HYDRAD_clean

### Setup Python environment
module load Anaconda3/5.0.0
source activate hic_simulation

### Configure Loop simulation
echo "Configuring initial conditions for loop$LOOP_NUM"
python $HOME/hi_c_simulation/configure_for_hydrad.py --loop_number $LOOP_NUM --interface_path $HOME/hi_c_simulation --ar_path $RESULTS_DIR/noaa12712_base --hydrad_path $HYDRAD_DIR --results_path $RESULTS_DIR/hydrodynamics
source deactivate

### Run HYDRAD
cd ${RESULTS_DIR}/hydrodynamics/loop${LOOP_NUM}/HYDRAD/
echo "Starting HYDRAD run for loop$LOOP_NUM on "`date`
srun HYDRAD.exe

### Stage-out results
# Run Steve's script here to prep results
