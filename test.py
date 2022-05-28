import torch
import torch.quantization.qconfig
from torch import optim
from generator_model import Generator
from discriminator_model import Discriminator
import config
from dataset import MapDataset
from torch.utils.data import DataLoader

from utils import test_some_examples, load_checkpoint


def test():

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    BCE = torch.nn.BCEWithLogitsLoss()

    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )

    test_some_examples(gen, disc, val_loader, epoch_num=1000, folder="evaluation")


test()