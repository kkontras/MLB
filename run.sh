#Please replace accordingly
#show.py shows the results
#train.py to run the training with --start_over if you want to overwrite previous checkpoints

###CREMA-D Res
python show.py --config ./configs/CREMA_D/release/res/unimodal_audio.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 
python show.py --config ./configs/CREMA_D/release/res/unimodal_video.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 
python show.py --config ./configs/CREMA_D/release/res/joint_training.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 
python show.py --config ./configs/CREMA_D/release/res/multiloss.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 
python show.py --config ./configs/CREMA_D/release/res/AGM.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 3.0
python show.py --config ./configs/CREMA_D/release/res/OGM.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 1.0
python show.py --config ./configs/CREMA_D/release/res/MSLR.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --kmepoch 20 --ilr_c 0.7 --ilr_g 1.3
python show.py --config ./configs/CREMA_D/release/res/MMCosine.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001  --mmcosine_scaling 10
python show.py --config ./configs/CREMA_D/release/res/MMPareto.json --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.0001 --wd 0.0001  --alpha 1.5
python show.py --config ./configs/CREMA_D/release/res/ReconBoost.json --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.0001 --wd 0.0001  --alpha 0.5 --recon_weight1 3 --recon_weight2 1 --recon_epochstages 1 --recon_ensemblestages 1
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode 2 --tanh_mode_beta 10

####CREMA-D Vit
python show.py --config ./configs/CREMA_D/release/vit/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.00005 --wd  5e-6
python show.py --config ./configs/CREMA_D/release/vit/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.00005 --wd  5e-6
python show.py --config ./configs/CREMA_D/release/vit/joint_training.json  --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd  5e-6
python show.py --config ./configs/CREMA_D/release/vit/multiloss.json  --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd  5e-6
python show.py --config ./configs/CREMA_D/release/vit/AGM.json  --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd  5e-6 --alpha 3.0
python show.py --config ./configs/CREMA_D/release/vit/OGM.json  --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd  5e-6 --alpha 1.0
python show.py --config ./configs/CREMA_D/release/vit/MSLR.json  --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd  5e-6
python show.py --config ./configs/CREMA_D/release/vit/MMCosine.json  --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd  5e-6
python show.py --config ./configs/CREMA_D/release/vit/MMPareto.json --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.00005 --wd  5e-6 --alpha 2.0 --batch_size 8
python show.py --config ./configs/CREMA_D/release/vit/ReconBoost.json --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.00005 --wd  5e-6 --alpha 0.5 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 4 --recon_ensemblestages 4 --batch_size 8
python show.py --config ./configs/CREMA_D/release/vit/MLB.json --default_config ./configs/CREMA_D/default_config_cremad.json   --lr 0.00005 --wd 5e-6 --alpha 5.0

###AVE Res
python show.py --config ./configs/AVE/release/res/unimodal_audio.json 
python show.py --config ./configs/AVE/release/res/unimodal_video.json 
python show.py --config ./configs/AVE/release/res/ens.json --default_config ./configs/AVE/default_config_ave_res.json 
python show.py --config ./configs/AVE/release/res/joint_training.json --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/multiloss.json --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/pre_frozen.json --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/pre_finetuned.json --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/MSLR.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --lr 0.001 --wd 0.0001 --kmepoch 10 --ilr_c 0.7 --ilr_g 1.3
python show.py --config ./configs/AVE/release/res/MMCosine.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --lr 0.001 --wd 0.0001 --mmcosine_scaling 1
python show.py --config ./configs/AVE/release/res/OGM.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --alpha 0.6 --lr 0.001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/AGM.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --alpha 3.0 --lr 0.0001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/MMPareto.json --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001 --alpha 0.5
python show.py --config ./configs/AVE/release/res/ReconBoost.json --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 1 --recon_ensemblestages 1
python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001 --alpha 2.0  --tanh_mode_beta 1

####UCF
python show.py --config ./configs/UCF/res/unimodal_audio.json --default_config ./configs/UCF/default_config_ucf.json --fold 1
python show.py --config ./configs/UCF/res/unimodal_video.json --default_config ./configs/UCF/default_config_ucf.json --fold 1
python show.py --config ./configs/UCF/res/ens.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001
python show.py --config ./configs/UCF/res/joint_training.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001
python show.py --config ./configs/UCF/res/multiloss.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001
python show.py --config ./configs/UCF/res/pre_frozen.json --default_config ./configs/UCF/default_config_ucf.json --fold 2 --lr 0.0001 --wd 0.0001 
python show.py --config ./configs/UCF/res/pre_finetuned.json --default_config ./configs/UCF/default_config_ucf.json --fold 2 --lr 0.0001 --wd 0.0001 
python show.py --config ./configs/UCF/res/MSLR.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001
python show.py --config ./configs/UCF/res/MMCosine.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001
python show.py --config ./configs/UCF/res/OGM.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 0.8 
python show.py --config ./configs/UCF/res/AGM.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 2.0 
python show.py --config ./configs/UCF/res/ReconBoost.json --default_config ./configs/UCF/default_config_ucf.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 4 --recon_ensemblestages 4
python show.py --config ./configs/UCF/res/MMPareto.json --default_config ./configs/UCF/default_config_ucf.json  --lr 0.001 --wd 0.0001 --alpha 3.0
python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 2.0  --tanh_mode_beta 0.5

##Sth-Sth
python show.py --config ./configs/SthSth/release/unimodal_video.json --default_config ./configs/SthSth/default_config_SthSth.json
python show.py --config ./configs/SthSth/release/unimodal_flow.json --default_config ./configs/SthSth/default_config_SthSth.json
python show.py --config ./configs/SthSth/release/joint_training.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json
python show.py --config ./configs/SthSth/release/multiloss.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json
python show.py --config ./configs/SthSth/release/OGM.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --alpha 1.0
python show.py --config ./configs/SthSth/release/AGM.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --alpha 1.0
python show.py --config ./configs/SthSth/release/ReconBoost.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json  --recon_weight1 5 --recon_weight2 1 --recon_epochstages 1 --recon_ensemblestages 1
python show.py --config ./configs/SthSth/release/MMPareto.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json  --alpha 3.0
python show.py --config ./configs/SthSth/release/MLB.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --alpha 2.0



#####Ablations#####

#Ablation on fusion gates/models
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode 2 --tanh_mode_beta 5 --cls nonlinear
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode 2 --tanh_mode_beta 10 --cls gated
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode 2 --tanh_mode_beta 20 --cls film
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode 2 --tanh_mode_beta 20 --cls tf


#Ablation study on balancing formulations and unimodal losses
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.8 --tanh_mode "ogm" --tanh_mode_beta 2 --cls linear_ogm_multi
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 3.0
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 4.0 --tanh_mode "all_linear" --tanh_mode_beta 2
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode "softmax" --tanh_mode_beta 2
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 5.0 --cls linear_stopgrad
python show.py --config ./configs/CREMA_D/release/res/MLB.json  --default_config ./configs/CREMA_D/default_config_cremad.json  --lr 0.001 --wd 0.0001 --alpha 5.0 --cls linear_ogm

python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001 --alpha 0.8 --tanh_mode "ogm" --tanh_mode_beta 2 --cls linear_ogm_multi
python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --alpha 2.0 --lr 0.001 --wd 0.0001
python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode "all_linear" --tanh_mode_beta 2
python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode "softmax" --tanh_mode_beta 2
python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --alpha 2.0 --lr 0.001 --wd 0.0001 --cls linear_stopgrad
python show.py --config ./configs/AVE/release/res/MLB.json  --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --alpha 0.5 --lr 0.001 --wd 0.0001 --cls linear_ogm

python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode "ogm" --tanh_mode_beta 2 --cls linear_ogm_multi
python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 2.0
python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 2.0 --tanh_mode "all_linear" --tanh_mode_beta 2
python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 2.0 --tanh_mode "softmax" --tanh_mode_beta 2
python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 0.5  --cls linear_stopgrad
python show.py --config ./configs/UCF/res/MLB.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --alpha 4.0  --cls linear_ogm


#Ablation study on balancing coefficient estimation method

python show.py --config ./configs/CREMA_D/release/res/MLB_ShapEq11.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 2 --lr 0.001 --wd 0.0001 --l 0.1 --batch_size 16 --alpha 5.0
python show.py --config ./configs/CREMA_D/release/res/MLB_PermEq11.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 2 --lr 0.001 --wd 0.0001 --l 0.1 --batch_size 16 --alpha 1.5
python show.py --config ./configs/CREMA_D/release/res/MLB_ShapEq4.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 2 --lr 0.001 --wd 0.0001 --l 0.1 --batch_size 16 --alpha 0.5
python show.py --config ./configs/CREMA_D/release/res/MLB_PermEq4.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 2 --lr 0.001 --wd 0.0001 --l 0.1 --batch_size 16 --alpha 1.5

python show.py --config ./configs/AVE/release/res/MLB_ShapEq11.json  --default_config ./configs/AVE/default_config_ave_res.json  --alpha 1.0 --lr 0.001 --wd 0.0001 --l 0.1
python show.py --config ./configs/AVE/release/res/MLB_PermEq11.json  --default_config ./configs/AVE/default_config_ave_res.json  --alpha 1.0 --lr 0.001 --wd 0.0001 --l 0.1
python show.py --config ./configs/AVE/release/res/MLB_ShapEq4.json  --default_config ./configs/AVE/default_config_ave_res.json  --alpha 3.0 --lr 0.001 --wd 0.0001 --l 0.1
python show.py --config ./configs/AVE/release/res/MLB_PermEq4.json  --default_config ./configs/AVE/default_config_ave_res.json  --alpha 1.0 --lr 0.001 --wd 0.0001 --l 0.1

python show.py --config ./configs/UCF/res/MLB_ShapEq11.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --l 0.1 --alpha 5.0
python show.py --config ./configs/UCF/res/MLB_PermEq11.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --l 0.1 --alpha 3.0
python show.py --config ./configs/UCF/res/MLB_ShapEq4.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --l 0.1 --alpha 1.0  --l 0.1 --batch_size 16
python show.py --config ./configs/UCF/res/MLB_PermEq4.json --default_config ./configs/UCF/default_config_ucf.json --fold 1  --lr 0.001 --wd 0.0001 --l 0.1 --alpha 2.0  --l 0.1 --batch_size 16



