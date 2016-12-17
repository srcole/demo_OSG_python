
#!/bin/bash
# load necessary modules
module load python/2.7
module load lapack
module load gcc
module load libgfortran
module load atlas
module load hdf5
module load stashcp

# transfer data from stashcache
stashcp /user/srcole/lfp_set/$2.npy data.npy

# untar and activate virtual environment
tar -xzf python_virtenv_demo.tar.gz
source ./python_virtenv_demo/bin/activate

# untar local library
tar -xzf misshapen.tar.gz

# Make directory for output files
mkdir out

# Run python script
./python_virtenv_demo/bin/python2.7 find_PsTs.py 

# tar output file
tar -czf out.$1.$2.tar.gz out

# Remove loaded data so not copied back
rm data.npy

# deactivate virtual environment
deactivate
