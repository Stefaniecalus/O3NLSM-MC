#!/bin/bash
#PBS -m a
#PBS -l walltime=60:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=16GB
#PBS -o $VSC_DATA/eos_wc/output/stdout.$PBS_JOBID
#PBS -e $VSC_DATA/eos_wc/Error/stderr.$PBS_JOBID

# Check if the 'venv' directory exists
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Activating..."
    # Activate the virtual environment
    source venv/bin/activate
else
    echo "Virtual environment 'venv' does not exist. Creating..."
    # Create a new virtual environment
    python3 -m venv venv
    # Activate the newly created virtual environment
    source venv/bin/activate
fi

# load Python module
module load Python

export I_MPI_COMPATIBILITY=1

# Move to proper working directory
cd $PBS_O_WORKDIR

# install requirements into virtual environment
pip install -r requirements.txt

# run the Python script
echo "Job started at : "`date`
python single_wcsweep.py --L $L --J $J --nsteps $nsteps --nth $nth --nspin $nspin --n $n --file $file
echo "Job ended at : "`date`
