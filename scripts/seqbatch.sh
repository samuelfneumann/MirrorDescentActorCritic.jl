#!/usr/bin/env bash


usage() {
    echo "Usage: $0 [-t DEP_TYPE] [-j :JOBID] jobscript [N]" 1>&2;
    echo "Sequentially schedule jobs using slurm" 1>&2;
}

jobid=""
offset=0
dep_type="afterany"
while getopts ":t:j:h" o; do
    case "${o}" in
	j)
	    jobid="${OPTARG}"
	    ;;
	t)
	    dep_type="${OPTARG}"
	    ;;
	h)
	    usage
	    exit 0
	    ;;
	:)
	    echo "option -${OPTARG} requires an argument"
	    exit 1
	    ;;
	?)
	    echo "invalid option -${OPTARG}"
	    exit 1
	    ;;
    esac
done

shift "$((OPTIND-1))"

# exit and print help message if invalid number of arguments given
if [[ $# = 0 ]] || [[ $# > 2 ]]; then
    usage
    exit 1
fi

# If job dependency was not set using -d, schedule a job and save its jobid
# as the first dependency
if [[ -z $jobid ]]; then
    sbatch_out=$(sbatch $1)
    sbatch_err=$?
    if [[ $sbatch_err != 0 ]]; then
	echo "could not schedule job ($sbatch_err): $sbatch_out"
	exit 2
    fi

    jobid=$(echo $sbatch_out | cut -d ' ' -f4)
    offset=1
    echo "Submitted batch job $jobid"
fi

for i in $(seq 1 $(( $2 - $offset ))); do
    sleep 2 # Don't overload the slurm scheduler
    old_jobid=$jobid
    sbatch_out=$(sbatch --dependency=$dep_type:$old_jobid $1 2>&1)

    sbatch_err=$?
    if [[ $sbatch_err != 0 ]]; then
	echo "could not schedule job ($sbatch_err): $sbatch_out"
	exit 3
    fi

    jobid=$(echo $sbatch_out | cut -d ' ' -f4)
    echo "Submitted batch job $jobid ($dep_type:$old_jobid)"
done
