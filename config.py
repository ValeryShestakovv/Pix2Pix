import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 24
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 10000
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_TEST = "gen_float32.pth.tar"
CHECKPOINT_DISC_TEST = "disc_float32.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)
transform_only_input = A.Compose(
    [

        A.Blur(blur_limit=7, p=0.6),
        A.ColorJitter(p=0.6),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
