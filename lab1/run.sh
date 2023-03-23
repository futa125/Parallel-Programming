#!/usr/bin/env bash

usage="$(basename "$0") -n <process_count>

where:
    -n number of processes to start"

options=':n:'
while getopts $options option
do
    case "$option" in
        n  ) process_count="$OPTARG"
             ;;
        \? ) echo "Unknown option: -$OPTARG" >&2
             echo "$usage" >&2
             exit 1
             ;;
        :  ) echo "Missing option argument for -$OPTARG" >&2
             echo "$usage" >&2
             exit 1
             ;;
        *  ) echo "Unimplemented option: -$option" >&2
             echo "$usage" >&2
             exit 1
             ;;
    esac
done

if ((OPTIND == 1))
then
    echo "No options specified" >&2
    echo "$usage" >&2
    exit 1
fi

mpirun -n "$process_count" python main.py
