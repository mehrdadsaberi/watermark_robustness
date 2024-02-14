from torch.utils.data import DataLoader
from utils import *
from torchvision.utils import save_image


import sys
import os
sys.path.append(os.path.join(os.getcwd(), "MBRS_repo"))

from network.Network import *

from MBRS_utils.load_test_setting import *

from tqdm import tqdm

class MBRS():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(H, W, message_length, noise_layers, self.device, batch_size, lr, with_diffusion)
        EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
        self.network.load_model_ed(EC_path)


    def run_MBRS(self, input_dataset=None, dataset_name="imagenet", out_dir='images/imagenet/MBRS'):

        network = self.network
        dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_fingerprinted_images = []

        gt_fingerprints  = "1110010111100001001110001101101011011100000011001100011010111110001100011011110110001110010101101011001011110101111101100110011110111001000100010101010011011111011001100000100111010010100000101111011110101011011010010111111011100111000111000000111101100011"
        fingerprint_size = len(gt_fingerprints)
        fingerprints = torch.zeros((batch_size, fingerprint_size), dtype=torch.float)
        for (i, fp) in enumerate(gt_fingerprints):
            fingerprints[:, i] = int(fp)
        fingerprints = fingerprints.to(self.device)


        for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):

            network.encoder_decoder.eval()
            network.discriminator.eval()

            with torch.no_grad():
                images, fingerprints = images.to(network.device), fingerprints.to(network.device)

                encoded_images = network.encoder_decoder.module.encoder(images, fingerprints)
                encoded_images = images + (encoded_images - images) * strength_factor
            
            all_fingerprinted_images.append(encoded_images.detach().cpu())

        
        
        all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()

        for idx in tqdm(range(len(all_fingerprinted_images))):
            image = all_fingerprinted_images[idx]
            filename = input_dataset.img_ids[idx]
            filename = filename.split('.')[0] + ".png"
            save_image(image, os.path.join(out_dir, f"{filename}"), padding=0)
            



    def decode_MBRS(self, input_dataset=None, dataset_name="imagenet"):

        network = self.network
        dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        bitwise_accuracies = []
        rev_error_rates = []

        gt_fingerprints  = "1110010111100001001110001101101011011100000011001100011010111110001100011011110110001110010101101011001011110101111101100110011110111001000100010101010011011111011001100000100111010010100000101111011110101011011010010111111011100111000111000000111101100011"
        fingerprint_size = len(gt_fingerprints)
        fingerprints = torch.zeros((batch_size, fingerprint_size), dtype=torch.float)
        for (i, fp) in enumerate(gt_fingerprints):
            fingerprints[:, i] = int(fp)
        fingerprints = fingerprints.to(self.device)

        # message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        for i, (encoded_images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            

            network.encoder_decoder.eval()
            network.discriminator.eval()

            with torch.no_grad():
                # use device to compute
                encoded_images, fingerprints = encoded_images.to(network.device), fingerprints.to(network.device)
                decoded_messages = network.encoder_decoder.module.decoder(encoded_images)
                

            # bit accuracy
            decoded_messages[decoded_messages > 0.5] = 1.
            decoded_messages[decoded_messages < 0.5] = 0.
            batch_accs = (fingerprints[: encoded_images.size(0)].detach() == decoded_messages[: encoded_images.size(0)]).float().mean(dim=1).cpu().numpy().tolist()
            bitwise_accuracies += batch_accs

            # loss
            for j in range(len(fingerprints)):
                rev_error_rates.append(1. - network.decoded_message_error_rate(fingerprints[j], decoded_messages[j]))

        # return bitwise_accuracies

        # MBRS has better performance with its own error rate, compared to rounding up the ouputs to 0,1 and calculating bit accuracy
        return rev_error_rates