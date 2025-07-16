import cv2
import math
import sys
import torch
import numpy as np
import argparse
from tqdm import trange
import os


sys.path.append('.')
import config as cfg
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder


# --- Helper functions (from your original script) ---
def _recursive_generator(frame1, frame2, down_scale, num_recursions, model, index):
    if num_recursions == 0:
        yield frame1, index
    else:
        mid_frame = model.hr_inference(frame1, frame2, True, TTA=TTA, fast_TTA=TTA, down_scale=down_scale)
        id = 2 ** (num_recursions - 1)
        yield from _recursive_generator(frame1, mid_frame, down_scale, num_recursions - 1, model, index - id)
        yield from _recursive_generator(mid_frame, frame2, down_scale, num_recursions - 1, model, index + id)

def make_inference(I0_, I2_, scale, n, model):
    return list(_recursive_generator(I0_, I2_, scale, int(math.log2(n)), model, n//2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='VFIMamba', type=str)
    parser.add_argument('--n', default=16, type=int) # Number of interpolated frames between two input frames
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--input_path', default='example', type=str, help='Path to input images folder')
    parser.add_argument('--output_path', default='example_output/', type=str, help='Path to save output images')

    args = parser.parse_args()
    assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'

    # Check CUDA and MPS
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    TTA = False
    if args.model == 'VFIMamba':
        TTA = True
        cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 32,
            depth = [2, 2, 2, 3, 3]
        )

    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()

    print(f"Using device: {device}")

    videogen = []
    image_extensions = ('.png', '.tif', '.jpg', '.jpeg', '.bmp', '.gif')
    matched_extension = None

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path '{args.input_path}' does not exist.")

    for f in os.listdir(args.input_path):
        for ext in image_extensions:
            if f.lower().endswith(ext):
                matched_extension = ext
                break
        if matched_extension and f.lower().endswith(matched_extension): 
            videogen.append(f)

    if not videogen:
        assert False, f"The input directory '{args.input_path}' does not contain valid image files."

    videogen.sort(key=lambda x: int(x.split('.')[0]))


    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output will be saved to: {args.output_path}")

    output_frame_count = 0

    pbar = trange(len(videogen) - 1, desc="Processing pairs", unit="pairs")

    for i in pbar:
        I0_filename = videogen[i]
        I2_filename = videogen[i+1]


        I0 = cv2.imread(os.path.join(args.input_path, I0_filename), cv2.IMREAD_UNCHANGED)
        I2 = cv2.imread(os.path.join(args.input_path, I2_filename), cv2.IMREAD_UNCHANGED)

        if len(I0.shape) == 2: 
            I0 = cv2.cvtColor(I0, cv2.COLOR_GRAY2BGR)
            I2 = cv2.cvtColor(I2, cv2.COLOR_GRAY2BGR)

        Is_dtype = I0.dtype 

        if Is_dtype == np.uint8:
            max_value = 255
        elif Is_dtype == np.uint16:
            max_value = 65535
        else:
            max_value = 1
        

        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / max_value).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / max_value).unsqueeze(0)
        
        padder = InputPadder(I0_.shape, divisor=32)
        I0_padded, I2_padded = padder.pad(I0_, I2_)

        all_frames_in_sequence = make_inference(I0_padded, I2_padded, args.scale, args.n, model)

        all_frames_in_sequence = sorted(all_frames_in_sequence, key = lambda x: x[1])

        output_filename = os.path.join(args.output_path, f'{output_frame_count:0>7d}{matched_extension}')
        cv2.imwrite(output_filename, I0)
        pbar.postfix = output_filename
        output_frame_count += 1
        for j, (pred_tensor, _) in enumerate(all_frames_in_sequence):
            pred = pred_tensor[0] 
            pred = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * max_value).astype(Is_dtype)
            

            output_filename = os.path.join(args.output_path, f'{output_frame_count:0>7d}{matched_extension}')
            cv2.imwrite(output_filename, pred)
            output_frame_count += 1
            pbar.postfix = output_filename
        output_filename = os.path.join(args.output_path, f'{output_frame_count:0>7d}{matched_extension}')
        cv2.imwrite(output_filename, I2)
        pbar.postfix = output_filename
        output_frame_count += 1


    print("Interpolation complete!")