# Sign Radar Classification - MICCAI 2025
Training arguments

--video_path
Type: Path
Description: Path to the dataset folder.

--split_path
Type: Path
Description: Path to the metadata file (in .pt format) of the dataset.

--load_embeddings
Type: int (converted to boolean)
Default: 0
Description: If using embeddings instead of videos (e.g., for classification with a frozen backbone).

--video_type
Type: str
Default: "mp4"
Description: Video type (e.g., mp4 or jpeg).

--multiCrop
Type: int (converted to boolean)
Options: 0, 1
Default: 0
Description: Flag to perform frame cropping.

--multiCropMode
Type: str
Default: "random"
Description: Cropping mode when multiCrop is enabled (e.g., "random" or "consecutives").

--frame_per_crop
Type: int
Default: 100
Description: Number of frames per crop in multiCrop mode.

--number_crop
Type: int
Default: 10
Description: Number of groups of frames (crops).

--resize
Type: int (nargs=3)
Default: [64, 128, 1024]
Description: Resize dimensions (height, width, depth/scale) for the videos.

--mean
Type: float (nargs=3)
Default: [0.0]
Description: Mean values for channel normalization.

--std
Type: float (nargs=3)
Default: [1.0]
Description: Standard deviation for channel normalization.

--num_fold
Type: int
Default: 0
Description: Test fold for nested cross-validation.

--inner_loop
Type: int
Default: 0
Description: Validation fold for nested cross-validation.

--task
Type: str
Default: "classification"
Description: Task to perform (either "reconstruction" or "classification").

--model
Type: str
Default: "endToEnd_Resnet"
Description: Model to be used. Options include models such as "autoencoder", "endToEnd", "endToEnd_fast", etc.

--num_classes
Type: int
Default: 126
Description: Number of classes in the dataset.

--use_rdm1, --use_rdm2, --use_rdm3
Type: int (converted to boolean)
Default: 1
Description: Flags to use the range doppler maps from respective receivers (rx1, rx2, rx3).

--use_rdmmti1, --use_rdmmti2, --use_rdmmti3
Type: int (converted to boolean)
Default: 1
Description: Flags to use the range doppler maps with MTI for each receiver.

--gradient_clipping_value
Type: int
Default: 0
Description: Gradient clipping value.

--optimizer
Type: str
Options: 'SGD', 'Adam', 'AdamW', 'RMSprop', 'LBFGS'
Default: "AdamW"
Description: Optimizer choice for training.

--learning_rate
Type: float
Default: 1e-4
Description: Learning rate.

--weight_decay
Type: float
Default: 1e-3
Description: L2 regularization weight.

--enable_scheduler
Type: int (converted to boolean)
Options: 0, 1
Default: 0
Description: Enables the learning rate scheduler.

--scheduler_factor
Type: float
Default: 8e-2
Description: Factor for reducing/increasing the learning rate via the scheduler.

--scheduler_threshold
Type: float
Default: 1e-2
Description: Threshold for updating the learning rate through the scheduler.

--scheduler_patience
Type: int
Default: 5
Description: Number of epochs to wait before changing the learning rate.

--batch_size
Type: int
Default: 16
Description: Batch size.

--epochs
Type: int
Default: 10000
Description: Maximum number of training epochs.

--experiment
Type: str
Default: None (if not provided, generated from the model name and timestamp)
Description: Name of the experiment.

--logdir
Type: str
Default: "./logs"
Description: Directory for logs and checkpoints.

--start_tensorboard_server
Type: int (converted to boolean)
Options: 0, 1
Default: 0
Description: Whether to start the Tensorboard server.

--tensorboard_port
Type: int
Default: 6006
Description: Port number for the Tensorboard server.

--saveLogsError, --saveLogs, --save_array_file, --save_image_file
Type: int (converted to boolean)
Description: Flags that control saving detailed error logs, general logs, array files, and image files respectively.

--enable_cudaAMP
Type: int (converted to boolean)
Options: 0, 1
Default: 1
Description: Enables CUDA Automatic Mixed Precision (AMP).

--device
Type: str
Default: "cuda"
Description: Specifies the device to use (e.g., "cpu", "cuda", "cuda:[number]").

--distributed
Type: int (converted to boolean)
Options: 0, 1
Default: 0
Description: Enables distributed training.

--dist_url
Type: str
Default: "env://"
Description: Initialization method for distributed training.