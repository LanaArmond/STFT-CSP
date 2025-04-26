#!/bin/bash

MaxParallelProcesses=6
PythonExecutable="python"

start_process() {
    local i=$1
    local j=$2
    echo -e "Iniciando script config_${i}_${j}.json em segundo plano"
    $PythonExecutable "$(dirname "$0")/kfold.py" jsons/config_2b_"$i"_"$j".json &
}

# Loop para executar scripts de 1 a 9
for ((i = 1; i <= 24; i++)); do
    for ((j = 1; j <= 9; j++)); do
        # Aguarda até que haja slots disponíveis
        while [ $(jobs -p | wc -l) -ge $MaxParallelProcesses ]; do
            # Aguarda 1 segundo antes de verificar novamente
            sleep 1
        done

        start_process $i $j
    done
done

wait

echo "Todos os scripts foram concluídos."
