# need to deactivate the hcoll since it gives errors otherwise...but even wihtout that it gives erors?
source wandbkey
source master_def
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 
#export CGX_COMPRESSION_BUCKET_SIZE=8192
export CGX_COMPRESSION_BUCKET_SIZE=1024
export CUDA_LAUNCH_BLOCKING=1
DUMMY_COMPRESSION=0
UNIFORM=1
ADAPTIVE=2
# SELECT number of bits
BITS=8

#export ANOMALY="YES"  uncomment to debug with detect anomaly
MPI_FLAGS='-v -np 3 -x PATH -x WANDB_API_KEY -x ANOMALY -x CGX_COMPRESSION_MODE -x CGX_QUANTIZATION_BITS -x CGX_COMPRESSION_BUCKET_SIZE -x CUDA_LAUNCH_BLOCKING --hostfile hostfile -mca coll ^hcoll --mca btl tcp,self --mca pml ob1'
BASE_COMMAND="python train_extraadam.py  --num-iter 500000 --default --model resnet --dist-backend cgx --cuda --wandb --fid-score --inception-score --quantization-bucket-size $CGX_COMPRESSION_BUCKET_SIZE --save-gen-samples --layernorm --batch-size 1024 --score-batch-size 4096 --master-addr ${MASTER_HOST}  --single-gpu-force --num-threads=5 --score-every 10"
NUQ_FLAGS="--nuq --warmup-milestones 0 200 1000 5000 --nuq-method=alq --nuq-every=10000 --quantization-bits $BITS "
UNIFORM_FLAGS="--quantization-bits $BITS "
FULL_PREC_FLAGS="--quantization-bits 32"

FULL_CMD="mpirun $MPI_FLAGS -- $BASE_COMMAND $FULL_PREC_FLAGS"
UNIFORM_CMD="mpirun $MPI_FLAGS -- $BASE_COMMAND $UNIFORM_FLAGS"
NUQ_CMD="mpirun $MPI_FLAGS -- $BASE_COMMAND $NUQ_FLAGS"

#SELECT the mode



#MODE=$DUMMY_COMPRESSION

MODE=$UNIFORM_
CMD=$UNIFORM_CMD
#
#MODE=$ADAPTIVE
#CMD=$NUQ_CMD

export CGX_COMPRESSION_MODE=$MODE
export CGX_QUANTIZATION_BITS=$BITS
echo "Mode ${MODE} (0=dummy,1=uniform,adaptive=2)"
echo "bits ${BITS}"
echo "Command:\n"
echo $CMD
$CMD
