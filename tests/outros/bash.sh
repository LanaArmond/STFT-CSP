#!/bin/bash

PythonExecutable="python"

# Loop para executar scripts de 1 a 9
for ((i = 1; i <= 48; i++)); do
    for ((j = 1; j <= 109; j++)); do
        if [ -f jsons/config_phy_"$i"_"$j".output ]; then
            echo -e "\njsons/config_phy_${i}_${j}.output já existe, pulando para o próximo"
            continue
        fi
        echo -e "\nIniciando script ${i} ${j} em segundo plano"
        $PythonExecutable "$(dirname "$0")/kfold.py" jsons/config_phy_"$i"_"$j".json
    done
done

echo "Todos os scripts foram concluídos."
