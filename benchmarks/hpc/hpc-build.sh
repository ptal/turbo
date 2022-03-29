module load CMake
module load CUDA
eb libxml2-2.9.10-GCCcore-11.2.0.eb --robot
module use ../.local/easybuild/modules/lib/libxml2 
module load libxml2
cd turbo/
./config.sh
cmake --build build/gpu-release/
cp build/gpu-release/turbo turbo