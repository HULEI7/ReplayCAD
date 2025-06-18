# [IJCAI 2025] 🔥ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection [[Paper](https://arxiv.org/abs/2505.06603)]


## 📅 News
- **[2025.06.18]** We release the training code.
- **[2025.05.10]** We release the [arXiv paper](https://arxiv.org/abs/2505.06603).
- **[2025.04.29]** 🎉🎉🎉 Our ReplayCAD is accepted by IJCAI 2025.

## 🔨 Requirement
```bash
conda create -n ReplayCAD python==3.8.19
conda activate ReplayCAD
pip install -r requirements.txt
```

## 🐳 Data

#### Mvtec

Download the MVTec-AD dataset from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move them to `data/mvtec/`.

#### Visa

Download the VisA dataset from [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Unzip the file and move them to `data/visa/`.


#### Generate data for replay

```bash
python get_replay_data.py
```

#### Get mask images

We provide the already generated mask image.

```bash
unzip SAM.zip
```

## 🍔 Pre-trained Model

### Mvtec

We use the pretrained diffusion model from [LDM](https://github.com/CompVis/latent-diffusion) repository, you can simply use the following command to obtain the pre-trained model.
```bash
wget -O textual_inversion-main/models/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

Please download bert-base-uncased from [here](https://huggingface.co/google-bert/bert-base-uncased), and put it in `textual_inversion-main/models/bert/bert-base-uncased`.

### Visa

For visa dataset, we use the pre-trained stable diffusion v1.5.

```bash
wget -O textual_inversion-main/models/v1-5-pruned.ckpt https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
```

Please download clip-vit-large-patch14 from [here](https://huggingface.co/openai/clip-vit-large-patch14), and put it in `textual_inversion-main/clip-vit-large-patch14`.


## 🚀 Training
###  Stage 1: Feature-guided data compression stage

Compress the data of each class into semantic and spatial features.

```bash
bash textual_inversion-main/tran_visa.sh
bash textual_inversion-main/tran_mvtec.sh
```

###  Stage 2: Replay-enhanced anomaly detection stage

Replay the data of historical classes by using compressed semantic and spatial features.

```bash
bash textual_inversion-main/generate_visa.sh
bash textual_inversion-main/generate_mvtec.sh
```

Put the generated data into the directory of the real data.

```bash
cp -r textual_inversion-main/output/visa/generate data/visa
cp -r textual_inversion-main/output/mvtec/generate data/mvtec

python get_visa_meta.py
python get_mvtec_meta.py
```

Train the anomaly detection model using replay data and real data together

```bash
bash run_visa.sh
bash run_mvtec.sh
```

###  FM score

After completing the above training, we trained the anomaly detection model using real data of all classes and obtained the FM score by calculating the difference from the above training results. Take mvtec as an example. To train with all real data, you just need to replace `data/mvtec/replay_meta.json` with `data/mvtec_meta.json`

## Citation
If you find our work inspiring or use our codebase in your research, please cite our work:
```
@article{hu2025replaycad,
  title={ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection},
  author={Hu, Lei and Gan, Zhiyong and Deng, Ling and Liang, Jinglin and Liang, Lingyu and Huang, Shuangping and Chen, Tianshui},
  journal={arXiv preprint arXiv:2505.06603},
  year={2025}
}
```

## ❤️ Acknowledgements
The research is partially supported by National Natural Science Foundation of China (No.62176093, 61673182), National Key Research and Development Program of China (No.2023YFC3502900), Key Realm Research and Development Program of Guangzhou (No.202206030001), Guangdong-Hong Kong-Macao Joint Innovation Project (No.2023A0505030016), Guangdong Emergency Management Science and Technology Program (No.2025YJKY001).

Our work is based on [ADer](https://github.com/zhangzjn/ADer), [textual_inversion](https://github.com/rinongal/textual_inversion) and [SAM](https://github.com/facebookresearch/segment-anything). Thank them for their excellent work.
