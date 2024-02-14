import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import random
from tqdm import tqdm
from torchvision import transforms
import cv2


BCH_POLYNOMIAL = 137
BCH_BITS = 5

def run_stega_stamp(dataset=None, dataset_name='imagenet', out_dir='images/imagenet/stegaStamp'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--secret', type=str)

    args = parser.parse_args(['--model', 'checkpoints/stegaStamp/stegastamp_pretrained',
                              '--save_dir', out_dir,
                              '--secret', 'Key10.3'])

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        for i, (img_tensor, label) in tqdm(enumerate(dataset)):
            image = img_tensor.numpy().transpose(1, 2, 0)
            image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_LINEAR)

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)


            im = Image.fromarray(np.array(rescaled))
            im.save(os.path.join(args.save_dir, f'{dataset.img_ids[i]}.png'))



def decode_stega_stamp(dataset=None, dataset_name='imagenet'):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--secret', type=str)

    args = parser.parse_args(['--model', 'checkpoints/stegaStamp/stegastamp_pretrained',
                              '--secret', 'Key10.3'])

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    
    secret_bits = []
    [[secret_bits.append((byte >> i) % 2) for i in range(7,-1,-1)] for byte in bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')]

    scores = []
    for i, (img_tensor, label) in tqdm(enumerate(dataset)):
        image = img_tensor.numpy().transpose(1, 2, 0)
        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_LINEAR)

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

        data_bits = []
        [[data_bits.append((byte >> i) % 2) for i in range(7,-1,-1)] for byte in data]



        if len(data_bits) != len(secret_bits):
            print("ERROR! Number of bits in decoded key is not as intended!")
            return []

        score = sum([data_bits[i] == secret_bits[i] for i in range(len(data_bits))]) / len(data_bits)
        scores.append(score)
    return scores

