# Sign Radar Classification - MICCAI 2025

This repository provides an end-to-end framework for training and evaluating radar-based sign language classification models, leveraging both video data and range Doppler maps. The approach presented here extends our previous studies and has been accepted for publication at MICCAI 2025.

Below is a concise list of critical training arguments:

• --video_path (Path): Path to the dataset folder.  
• --split_path (Path): Path to the dataset metadata (.pt file).  
• --load_embeddings (bool): Use precomputed embeddings instead of raw video frames.  
• --video_type (str): Type of video input (e.g., "mp4" or "jpeg").  
• --multiCrop (bool): Enable multiple cropping strategies for temporal frames.  
• --multiCropMode (str): Mode of cropping ("random" or "consecutives").  
• --frame_per_crop (int): Number of frames per crop in multiCrop mode.  
• --number_crop (int): Number of crop groups.  
• --resize (int[3]): Resize dimensions (height, width, depth/scale).  
• --mean (float[3]): Mean for each channel for normalization.  
• --std (float[3]): Standard deviation for each channel.  
• --num_fold (int): Test fold for nested cross-validation.  
• --inner_loop (int): Validation fold for nested cross-validation.  
• --task (str): Task to execute ("classification" or "reconstruction").  
• --model (str): Model name (e.g., "endToEnd", "autoencoder", "fast").  
• --num_classes (int): Total number of classes.  
• --use_rdm1, --use_rdm2, --use_rdm3 (bool): Use range Doppler maps from respective receivers.  
• --use_rdmmti1, --use_rdmmti2, --use_rdmmti3 (bool): Use range Doppler maps with MTI from each receiver.  
• --gradient_clipping_value (int): Gradient clipping parameter.  
• --optimizer (str): Choice of optimizer ("SGD", "Adam", "AdamW", etc.).  
• --learning_rate (float): Learning rate.  
• --weight_decay (float): L2 regularization weight.  
• --enable_scheduler (bool): Enable learning rate scheduler.  
• --scheduler_factor (float): Factor for learning rate changes.  
• --scheduler_threshold (float): Threshold for triggering scheduler updates.  
• --scheduler_patience (int): Patience (epochs) before scheduler modifies the learning rate.  
• --batch_size (int): Batch size per training iteration.  
• --epochs (int): Maximum number of training epochs.  
• --experiment (str): Experiment name (default auto-generated if not supplied).  
• --logdir (str): Directory for logging and checkpoints.  
• --start_tensorboard_server (bool): Automatically launch a TensorBoard server.  
• --tensorboard_port (int): Port for the TensorBoard server.  
• --saveLogsError, --saveLogs, --save_array_file, --save_image_file (bool): Flags for saving logs/files for error analysis, general predictions, arrays, and images.  
• --enable_cudaAMP (bool): Enable CUDA Automatic Mixed Precision.  
• --device (str): Device to use (e.g., "cpu", "cuda", "cuda:0").  
• --distributed (bool): Enable distributed training.  
• --dist_url (str): Initialization method for distributed training.

If you make use of this code, please cite us:

@inproceedings{mineo2024sign,  
  title={Sign Language Recognition for Patient-Doctor Communication: A Multimedia/Multimodal Dataset},  
  author={Mineo, Raffaele and Caligiore, Gaia and Spampinato, Concetto and Fontana, Sabina and Palazzo, Simone and Ragonese, Egidio},  
  booktitle={2024 IEEE 8th Forum on Research and Technologies for Society and Industry Innovation (RTSI)},  
  pages={202--207},  
  year={2024},  
  organization={IEEE}  
}

@inproceedings{caligiore-etal-2024-multisource,  
  title = {Multisource Approaches to Italian Sign Language ({LIS}) Recognition: Insights from the MultiMedaLIS Dataset},  
  author = {Caligiore, Gaia and Mineo, Raffaele and Spampinato, Concetto and Ragonese, Egidio and Palazzo, Simone and Fontana, Sabina},  
  editor = {DellOrletta, Felice and Lenci, Alessandro and Montemagni, Simonetta and Sprugnoli, Rachele},  
  booktitle = {Proceedings of the 10th Italian Conference on Computational Linguistics (CLiC-it 2024)},  
  month = dec,  
  year = {2024},  
  address = {Pisa, Italy},  
  publisher = {CEUR Workshop Proceedings},  
  url = {https://aclanthology.org/2024.clicit-1.17/},  
  pages = {132--140},  
  ISBN = {979-12-210-7060-6}  
}

@inproceedings{mineo2025radar,  
  title={Radar-Based Imaging for Sign Language Recognition in Medical Communication},  
  author={Mineo, Raffaele and Caligiore, Gaia and Proietto Salanitri, Federica and Kavasidis, Isaak and Polikovsky, Senya and Fontana, Sabina and Ragonese, Egidio and Spampinato, Concetto and Palazzo, Simone},  
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},  
  year={2025},  
  organization={Springer}  
}

This code is taken from https://github.com/IngRaffaeleMineo/3D-BCPTcode and modified to our target.