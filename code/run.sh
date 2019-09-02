#!/usr/bin/env bash

while getopts "m:g:d:s:l:" opt;
do
    case ${opt} in
        m) mode=$OPTARG ;;
        d) data=$OPTARG ;;
    esac
done

log_folder=../log/
if [ ! -d ${log_folder} ];then
    mkdir -p ${log_folder}
    echo "create log folder: " ${log_folder}
fi
echo "log folder: " ${log_folder}

echo "training data: " ${data}

data_split=(${data//// })
echo ${data_split}
log_file=${log_folder}${data_split[-1]}


cur_time="`date +%Y%m%d%H%M%S`"
if [ ${mode} == "ITC" ]; then
    log_file=${log_file}_ITC_${cur_time}.log
    echo "log file: " ${log_file}
    python3 run_ITC.py --training_data ${data}/ >> ${log_file}
elif [ ${mode} == "SSL" ]; then
    log_file=${log_file}_SSL_${cur_time}.log
    echo "log file: " ${log_file}
    python3 run_SSL.py --training_data ${data}/ >> ${log_file}
fi

