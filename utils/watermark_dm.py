import argparse
import os
import glob
import PIL


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from utils.utils import CustomImageFolder


def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


def load_data():
    global dataset, dataloader

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
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from WatermarkDM.string2img.models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    gt_fingerprints  = "0100010001000010111010111111110011101000001111101101010110000001"
    fingerprint_size = len(gt_fingerprints)
    fingerprints = torch.zeros((BATCH_SIZE, fingerprint_size), dtype=torch.float)
    for (i, fp) in enumerate(gt_fingerprints):
        fingerprints[:, i] = int(fp)
    fingerprints = fingerprints.cuda()
    # fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
    # fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE)
    # fingerprints = fingerprints.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)

    bitwise_accuracy = 0

    for images, _ in tqdm(dataloader):

        # generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)
            fingerprints = fingerprints.to(device)

        images = images.to(device)

        fingerprinted_images = HideNet(fingerprints[: images.size(0)], images)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()

    dirname = args.output_dir
    # if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
    #     os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()

    print("Saving fingerprinted images...")
    f = open(os.path.join(args.output_dir_note, "embedded_fingerprints.txt"), "w")
    for idx in tqdm(range(len(all_fingerprinted_images))):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        filename = dataset.img_ids[idx]
        filename = filename.split('.')[0] + ".png"
        # save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{filename}"), padding=0)
        save_image(image, os.path.join(args.output_dir, f"{filename}"), padding=0)
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")

        save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        save_image(torch.abs(images - fingerprinted_images)[:49], os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)


def run_watermark_dm(input_dataset=None, dataset_name="imagenet", out_dir='images/imagenet/watermarkDM'):

    global args, device, uniform_rv, BATCH_SIZE, dataset

    dataset = input_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
    parser.add_argument(
        "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
    )
    parser.add_argument("--data_dir", type=str, help="Directory with images.")
    parser.add_argument(
        "--output_dir", type=str, help="Path to save watermarked images to."
    )
    parser.add_argument(
        "--image_resolution", type=int, help="Height and width of square images."
    )
    parser.add_argument(
        "--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate fingerprint detection accuracy."
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
    parser.add_argument("--cuda", type=int, default=0)

    args = parser.parse_args(['--encoder_path', f'checkpoints/watermarkDM/{dataset_name}_encoder.pth',
                '--decoder_path', f'checkpoints/watermarkDM/{dataset_name}_decoder.pth',
                '--image_resolution', str(32 if dataset_name == 'cifar10' else 256),
                '--identical_fingerprints', '--batch_size', '128'])
    BATCH_SIZE = args.batch_size


    uniform_rv = torch.distributions.uniform.Uniform(
        torch.tensor([0.0]), torch.tensor([1.0])
    )
    
    if int(args.cuda) == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    root_data_dir = ""
    image_outdir  = out_dir
    note_outdir   = out_dir
    
    # process cifar10 dataset
    
    args.data_dir         = root_data_dir
    args.output_dir       = image_outdir
    args.output_dir_note  = note_outdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir_note):
        os.makedirs(args.output_dir_note)

    if not dataset:
        load_data()
    load_models()
    embed_fingerprints()




def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from WatermarkDM.string2img.models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path)
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.image_resolution, 3, FINGERPRINT_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)


def extract_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []
    bitwise_accuracies = []

    BATCH_SIZE = args.batch_size
    
    # transform gt_fingerprints to 
    gt_fingerprints  = "0100010001000010111010111111110011101000001111101101010110000001"
    fingerprint_size = len(gt_fingerprints)
    z = torch.zeros((args.batch_size, fingerprint_size), dtype=torch.float)
    for (i, fp) in enumerate(gt_fingerprints):
        z[:, i] = int(fp)
    z = z.cuda()


    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    for images, _ in tqdm(dataloader):
        images = images.to(device)

        fingerprints = RevealNet(images)
        fingerprints = (fingerprints > 0).long()
        
        batch_accs = (fingerprints[: images.size(0)].detach() == z[: images.size(0)]).float().mean(dim=1).cpu().numpy().tolist()
        bitwise_accuracies += batch_accs

        all_fingerprinted_images.append(images.detach().cpu())
        all_fingerprints.append(fingerprints.detach().cpu())

    # dirname = args.output_dir
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    return bitwise_accuracies
    # bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
    # print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}") # non-corrected
          
    # write in file
    # f = open(os.path.join(args.output_dir, "detected_fingerprints.txt"), "w")
    # for idx in range(len(all_fingerprints)):
    #     fingerprint = all_fingerprints[idx]
    #     fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
    #     _, filename = os.path.split(dataset.filenames[idx])
    #     filename = filename.split('.')[0] + ".png"
    #     f.write(f"{filename} {fingerprint_str}\n")
    # f.close()



def decode_watermark_dm(input_dataset=None, dataset_name="imagenet"):

    global args, device, dataset

    dataset = input_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory with images.")
    parser.add_argument(
        "--output_dir", type=str, help="Path to save watermarked images to."
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        help="Height and width of square images.",
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        help="Path to trained StegaStamp decoder.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--check", action="store_true", help="Validate fingerprint detection accuracy."
    )

    args = parser.parse_args(['--decoder_path', f'checkpoints/watermarkDM/{dataset_name}_decoder.pth',
                '--image_resolution', str(32 if dataset_name == 'cifar10' else 256),
                '--batch_size', '128',
                '--data_dir', f"images/{dataset_name}/watermarkDM",
                '--output_dir', f"results/logs/watermark_dm_notes/{dataset_name}/"])

    if int(args.cuda) == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    load_decoder()
    if not dataset:
        load_data()
    return extract_fingerprints()

