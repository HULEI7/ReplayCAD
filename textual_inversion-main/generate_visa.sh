#change  "embedding_path", "class_layer_path" to yours

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/pcb12024-09-14T15-46-42_SD_visa_addmask/checkpoints/embeddings_gs-26499.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/pcb1 \
--class_layer_path textual_inversion-main/logs/pcb12024-09-14T15-46-42_SD_visa_addmask/checkpoints/mask_linear-26499.pt \
--conference_mask_path SAM/data/visa_conference/pcb1

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/pcb22024-09-14T22-45-04_SD_visa_addmask/checkpoints/embeddings_gs-29999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/pcb2 \
--class_layer_path textual_inversion-main/logs/pcb22024-09-14T22-45-04_SD_visa_addmask/checkpoints/mask_linear-29999.pt \
--conference_mask_path SAM/data/visa_conference/pcb2

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/pcb32024-09-15T06-32-20_SD_visa_addmask/checkpoints/embeddings_gs-29499.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/pcb3 \
--class_layer_path textual_inversion-main/logs/pcb32024-09-15T06-32-20_SD_visa_addmask/checkpoints/mask_linear-29499.pt \
--conference_mask_path SAM/data/visa_conference/pcb3

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/pcb42024-09-16T14-44-42_SD_visa_addmask/checkpoints/embeddings_gs-29999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/pcb4 \
--class_layer_path textual_inversion-main/logs/pcb42024-09-16T14-44-42_SD_visa_addmask/checkpoints/mask_linear-29999.pt \
--conference_mask_path SAM/data/visa_conference/pcb4

CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 100 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/macaroni12024-09-13T14-24-41_LD_visa_addmask/checkpoints/embeddings_gs-9999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/macaroni1 \
--class_layer_path textual_inversion-main/logs/macaroni12024-09-13T14-24-41_LD_visa_addmask/checkpoints/mask_linear-9999.pt \
--conference_mask_path SAM/data/visa_conference/macaroni1

CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 50 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/macaroni22024-08-04T11-09-23_LD_visa/checkpoints/embeddings_gs-14535.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/macaroni2

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/capsules2024-09-20T09-08-27_SD_visa/checkpoints/embeddings_gs-52499.pt \
--prompt "a photo of *" --config textual_inversion-main/configs/stable-diffusion/v1-inference.yaml \
--outdir textual_inversion-main/output/visa/generate/capsules

CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 100 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/candle2024-09-12T20-48-51_LD_visa_addmask/checkpoints/embeddings_gs-9999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/candle \
--class_layer_path textual_inversion-main/logs/candle2024-09-12T20-48-51_LD_visa_addmask/checkpoints/mask_linear-9999.pt \
--conference_mask_path SAM/data/visa_conference/candle

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/cashew2024-09-17T23-00-07_SD_visa_addmask/checkpoints/embeddings_gs-29999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/cashew \
--class_layer_path textual_inversion-main/logs/cashew2024-09-17T23-00-07_SD_visa_addmask/checkpoints/mask_linear-29999.pt \
--conference_mask_path SAM/data/visa_conference/cashew

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/chewinggum2024-09-18T06-06-03_SD_visa_addmask/checkpoints/embeddings_gs-29999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/chewinggum \
--class_layer_path textual_inversion-main/logs/chewinggum2024-09-18T06-06-03_SD_visa_addmask/checkpoints/mask_linear-29999.pt \
--conference_mask_path SAM/data/visa_conference/chewinggum

CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img_with_mask.py --ddim_eta 0.0 --n_samples 2 --n_iter 200 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/fryum2024-09-19T15-27-27_SD_visa_addmask/checkpoints/embeddings_gs-31656.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/visa/generate/fryum \
--class_layer_path textual_inversion-main/logs/fryum2024-09-19T15-27-27_SD_visa_addmask/checkpoints/mask_linear-31656.pt \
--conference_mask_path SAM/data/visa_conference/fryum