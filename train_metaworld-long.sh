## 0 6 9 20  24 23 25

seeds=(6)
game_list=(
    'door-close'
    'button-press'
    'window-close'
    'handle-press'
    'drawer-close'
    'button-press-topdown'
    )
    
    
for env_name in "${game_list[@]}"; 
do
    for seed in "${seeds[@]}";
    do
        base_model=Mamba
        model_version=1_2_3
        cuda_device=0
        sample=normal
        suite=metaworld

        python -u train_metaworld.py \
            -suite ${suite} \
            -env_name ${env_name} \
            -seed ${seed} \
            -base_model ${base_model} \
            -version ${model_version} \
            -config_path "config_files/${base_model}.yaml" \

        echo $env_name
        echo $seed

    done
done


    


