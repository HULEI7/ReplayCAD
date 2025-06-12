#change  "embedding_path", "class_layer_path" to yours


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 100 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/screw2024-09-08T22-26-05_LD_mvtec_addmask/checkpoints/embeddings_gs-19999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/screw \
--class_layer_path textual_inversion-main/logs/screw2024-09-08T22-26-05_LD_mvtec_addmask/checkpoints/mask_linear-19999.pt \
--conference_mask_path SAM/data/mvtec_conference/screw

CUDA_VISIBLE_DEVICES=1 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 100 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/metal_nut2024-09-08T22-27-26_LD_mvtec_addmask/checkpoints/embeddings_gs-19999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/metal_nut \
--class_layer_path textual_inversion-main/logs/metal_nut2024-09-08T22-27-26_LD_mvtec_addmask/checkpoints/mask_linear-19999.pt \
--conference_mask_path SAM/data/mvtec_conference/metal_nut

CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 100 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/transistor2024-09-08T03-00-10_LD_mvtec_addmask/checkpoints/embeddings_gs-4999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/transistor \
--class_layer_path textual_inversion-main/logs/transistor2024-09-08T03-00-10_LD_mvtec_addmask/checkpoints/mask_linear-4999.pt \
--conference_mask_path SAM/data/mvtec_conference/transistor

CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 100 --scale 5.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/grid2024-09-11T10-47-46_LD_mvtec_addmask/checkpoints/embeddings_gs-24999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/grid \
--class_layer_path textual_inversion-main/logs/grid2024-09-11T10-47-46_LD_mvtec_addmask/checkpoints/mask_linear-24999.pt \
--conference_mask_path SAM/data/mvtec_conference/grid


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/bottle2024-11-17T12-01-43_LD_mvtec_addmask/checkpoints/embeddings_gs-2499.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/bottle \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/bottle2024-11-17T12-01-43_LD_mvtec_addmask/checkpoints/mask_linear-2499.pt \
--conference_mask_path SAM/data/mvtec_conference/bottle


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 50 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/cable2024-11-17T13-53-28_LD_mvtec_addmask/checkpoints/embeddings_gs-5999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/cable \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/cable2024-11-17T13-53-28_LD_mvtec_addmask/checkpoints/mask_linear-5999.pt \
--conference_mask_path SAM/data/mvtec_conference/cable


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/capsule2024-11-17T22-31-57_LD_mvtec_addmask/checkpoints/embeddings_gs-14999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/capsule \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/capsule2024-11-17T22-31-57_LD_mvtec_addmask/checkpoints/mask_linear-14999.pt \
--conference_mask_path SAM/data/mvtec_conference/capsule


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 1.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/carpet2024-11-18T10-15-07_LD_mvtec_addmask/checkpoints/embeddings_gs-9999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/carpet \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/carpet2024-11-18T10-15-07_LD_mvtec_addmask/checkpoints/mask_linear-9999.pt \
--conference_mask_path SAM/data/mvtec_conference/carpet

CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/hazelnut2024-11-17T22-33-26_LD_mvtec_addmask/checkpoints/embeddings_gs-19999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/hazelnut \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/hazelnut2024-11-17T22-33-26_LD_mvtec_addmask/checkpoints/mask_linear-19999.pt \
--conference_mask_path SAM/data/mvtec_conference/hazelnut


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 1.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/leather2024-11-17T21-22-04_LD_mvtec_addmask/checkpoints/embeddings_gs-3499.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/leather \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/leather2024-11-17T21-22-04_LD_mvtec_addmask/checkpoints/mask_linear-3499.pt \
--conference_mask_path SAM/data/mvtec_conference/leather


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/pill2024-11-17T15-45-16_LD_mvtec_addmask/checkpoints/embeddings_gs-1499.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/pill \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/pill2024-11-17T15-45-16_LD_mvtec_addmask/checkpoints/mask_linear-1499.pt \
--conference_mask_path SAM/data/mvtec_conference/pill


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/tile2024-11-18T16-19-05_LD_mvtec_addmask/checkpoints/embeddings_gs-4999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/tile \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/tile2024-11-18T16-19-05_LD_mvtec_addmask/checkpoints/mask_linear-4999.pt \
--conference_mask_path SAM/data/mvtec_conference/tile


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 10.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/toothbrush2024-11-17T17-37-18_LD_mvtec_addmask/checkpoints/embeddings_gs-1499.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/toothbrush \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/toothbrush2024-11-17T17-37-18_LD_mvtec_addmask/checkpoints/mask_linear-1499.pt \
--conference_mask_path SAM/data/mvtec_conference/toothbrush


CUDA_VISIBLE_DEVICES=0 python scripts/txt2img_with_mask.py --ddim_eta 0.0 --n_samples 8 --n_iter 25 --scale 1.0 --ddim_steps 50 \
--embedding_path textual_inversion-main/logs/mvtec_add_mask_2/wood2024-11-17T19-30-32_LD_mvtec_addmask/checkpoints/embeddings_gs-5999.pt \
--prompt "a photo of *" --outdir textual_inversion-main/output/mvtec/generate/wood \
--class_layer_path textual_inversion-main/logs/mvtec_add_mask_2/wood2024-11-17T19-30-32_LD_mvtec_addmask/checkpoints/mask_linear-5999.pt \
--conference_mask_path SAM/data/mvtec_conference/wood

