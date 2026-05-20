
seeds=(0)
game_list=(
    # 'Breakout'
    # 'BankHeist'
    # 'UpNDown'
    # 'Frostbite'
    # 'BattleZone'
    # 'Freeway'
    # 'BankHeist'
    'RoadRunner'
    )
    
    
for env_name in "${game_list[@]}"; 
do
    for seed in "${seeds[@]}";
    do
        base_model=Glam
        model_version=7726
        cuda_device=0
        sample=normal
        suite=atari

        python -u train2.py \
            -suite ${suite} \
            -env_name ${env_name} \
            -seed ${seed} \
            -base_model ${base_model} \
            -version ${model_version} \
            -config_path "config_files/${base_model}.yaml" \

        
        echo $env_name
        echo $seed

        torch_seeds=($seed)
        for torch_seeds in "${torch_seeds[@]}";
        do
            echo $env_name
            ckpt_path="/home/hq/LSTW/MSTORM_base/data/ckpt/${env_name}_seed${seed}_${base_model}_${model_num}"
            python -u eval.py \
                -env_name "ALE/${env_name}-v5" \
                -game "${env_name}" \
                -run_name "${torch_seeds}${env_name}-${base_model}-seed${seed}" \
                -base_model ${base_model} \
                -num ${model_num} \
                -config_path "${ckpt_path}/config.yaml" \
                -ckpt_path ${ckpt_path} \
                -seed ${torch_seeds} 
        done

    done
done

