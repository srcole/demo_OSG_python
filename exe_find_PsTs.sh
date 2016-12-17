
#!/bin/bash
module load python/2.7
module load lapack
module load gcc
module load libgfortran
module load atlas
module load hdf5

# untar virtual environment
tar -xzf my_virtenv.tar.gz
source ./my_virtenv/bin/activate

# untar local libraries
tar -xzf misshapen.tar.gz
tar -xzf tools.tar.gz
tar -xzf processed.tar.gz
ls -l 
echo $LD_LIBRARY_PATH
# Run the python script 
./my_virtenv/bin/python2.7 calcgamma_v1.py 

mkdir -p processed_out.$1.$2
mv ./processed/*.npy  processed_out.$1.$2/.
tar -czf processed_out.$1.$2.tar.gz processed_out.$1.$2

 deactivate
