#!/bin/bash

### SLURM Parameters
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=hi-c_hydrad
#SBATCH --time=07:59:59
#SBATCH --mail-user=wtb2@rice.edu
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

### Setup needed parameters
printf -v LOOP_NUM "%06d" $SLURM_ARRAY_TASK_ID
RESULTS_DIR=$SHARED_SCRATCH/wtb2/hi_c_simulation
HYDRAD_DIR=$SHARED_SCRATCH/wtb2/hi_c_simulation/HYDRAD_clean
RUN_DIR=${RESULTS_DIR}/hydrodynamics/loop${LOOP_NUM}
REDUCED_FILENAME=${RESULTS_DIR}/hydrodynamics/reduced_results/loop${LOOP_NUM}_uniform.h5

### Setup Python environment
module load Anaconda3/5.0.0
source activate hic_simulation

### Configure Loop simulation
rm -rf $RUN_DIR
echo "Configuring initial conditions for loop$LOOP_NUM"
srun python $HOME/hi_c_simulation/configure_for_hydrad.py --loop_number $LOOP_NUM --interface_path $HOME/hi_c_simulation --ar_path $RESULTS_DIR/noaa12712_base --hydrad_path $HYDRAD_DIR --results_path $RESULTS_DIR/hydrodynamics
# Copy config file to separate location
cp $RUN_DIR/hydrad_tools_config.asdf $RESULTS_DIR/hydrodynamics/config_files/loop$LOOP_NUM.hydrad_config.asdf

### Run HYDRAD
cd $RUN_DIR
echo "Starting HYDRAD run for loop$LOOP_NUM on "`date`
srun ${RUN_DIR}/HYDRAD.exe >> ${RUN_DIR}/job_status.out

### Reduce data to uniform grid
rm $REDUCED_FILENAME
echo "Reducing HYDRAD data for loop$LOOP_NUM"
srun python $HOME/hi_c_simulation/make_uniform_grid.py --hydrad_root $RUN_DIR --output_file $REDUCED_FILENAME

### Deactivate Python environment
source deactivate
