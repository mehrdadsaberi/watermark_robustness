import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory with image dataset."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save results to."
)
parser.add_argument(
    "--bit_length",
    type=int,
    default=64,
    required=True,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=32,
    required=True,
    help="Height and width of square images.",
)
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)

parser.add_argument(
    "--l2_loss_await",
    help="Train without L2 loss for the first x iterations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--l2_loss_weight",
    type=float,
    default=10,
    help="L2 loss weight for image fidelity.",
)
parser.add_argument(
    "--l2_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase L2 loss weight over x iterations.",
)

parser.add_argument(
    "--BCE_loss_weight",
    type=float,
    default=1,
    help="BCE loss weight for fingerprint reconstruction.",
)

args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

import models


LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)


def generate_random_fingerprints(bit_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, bit_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        for i in range(50):
            if i == 0:
                self.filenames = glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.png"))
                self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpeg")))
                self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpg")))
            else:
                self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.png")))
                self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpeg")))
                self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    SECRET_SIZE = args.bit_length

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_RESOLUTION),
                transforms.CenterCrop(IMAGE_RESOLUTION),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"

    device = torch.device("cuda")

    load_data()
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )

    global_step = 0
    steps_since_l2_loss_activated = -1

    for i_epoch in range(args.num_epochs):
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
        )
        for images, _ in tqdm(dataloader):
            global_step += 1

            batch_size = min(args.batch_size, images.size(0))
            fingerprints = generate_random_fingerprints(
                args.bit_length, batch_size, (args.image_resolution, args.image_resolution)
            )

            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)

            fingerprinted_images = encoder(fingerprints, clean_images)
            residual = fingerprinted_images - clean_images

            decoder_output = decoder(fingerprinted_images)

            criterion = nn.MSELoss()
            l2_loss = criterion(fingerprinted_images, clean_images)

            criterion = nn.BCEWithLogitsLoss()
            BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))

            loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1

            # Logging
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step),
                print("Bitwise accuracy {}".format(bitwise_accuracy))
                writer.add_scalar("loss", loss, global_step),
                writer.add_scalar("BCE_loss", BCE_loss, global_step),
                writer.add_scalars(
                    "clean_statistics",
                    {"min": clean_images.min(), "max": clean_images.max()},
                    global_step,
                ),
                writer.add_scalars(
                    "with_fingerprint_statistics",
                    {
                        "min": fingerprinted_images.min(),
                        "max": fingerprinted_images.max(),
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "residual_statistics",
                    {
                        "min": residual.min(),
                        "max": residual.max(),
                        "mean_abs": residual.abs().mean(),
                    },
                    global_step,
                ),
                print(
                    "residual_statistics: {}".format(
                        {
                            "min": residual.min(),
                            "max": residual.max(),
                            "mean_abs": residual.abs().mean(),
                        }
                    )
                )
                writer.add_image(
                    "clean_image", make_grid(clean_images, normalize=True), global_step
                )
                writer.add_image(
                    "residual",
                    make_grid(residual, normalize=True, scale_each=True),
                    global_step,
                )
                writer.add_image(
                    "image_with_fingerprint",
                    make_grid(fingerprinted_images, normalize=True),
                    global_step,
                )
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )

                writer.add_scalar(
                    "loss_weights/l2_loss_weight", l2_loss_weight, global_step
                )
                writer.add_scalar(
                    "loss_weights/BCE_loss_weight",
                    BCE_loss_weight,
                    global_step,
                )

            # checkpointing
            if global_step % 5000 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                f = open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w")
                f.write(str(global_step))
                f.close()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
