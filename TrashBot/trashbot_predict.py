from huggingface_hub import login
from torch import nn
import numpy as np
from PIL import Image, ImageFilter
from datasets import load_dataset
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.colors as colors
from colour import Color

def trashbot_predict(dataset, model, blur_factor=3):

    # Log in to huggingface hub
    login("TOKEN GOES HERE")
    hf_username = "USERNAME GOES HERE"
    hub_model_id = model
    dataset_name = dataset

    # get model and feature extractor
    model_checkpoint = "nvidia/mit-b5"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(f"{hf_username}/{hub_model_id}")

    # Load dataset from local folder
    ds = load_dataset("imagefolder", data_dir=f"../drone_photos/{dataset_name}/JPG")

    # Sigmoid function to convert logit tensor to probabilities
    def sigmoid(tensor):
        return 1 / (1 + np.exp(-1*tensor.detach().numpy()))

    # Function to make segmentation overlay
    def get_seg_overlay(image, seg, proba):

        # Initialize color segment bitmap
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3  
        
        # Function to create a colormap
        def make_cmap( ramp_colors ):         
            color_ramp = colors.LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in ramp_colors ] )    
            return color_ramp

        # Create colormap and apply to probabilities
        cmap = make_cmap(['#4F4F4F','#606060','#DEFF00','#DEFF00','#DEFF00']) # Highlighter
        # cmap = make_cmap(['#4F4F4F','#606060','#ffef14','#ffef14','#ffef14']) # Gold
        color_seg = cmap(proba)

        # Opacity maps for low, med, high probability pixels
        alpha_mask_low = proba < 0.4  
        alpha_mask_med_low = (proba > 0.4) & (proba < 0.5) 
        alphas = np.ones(shape=proba.shape)
        alphas = proba * 0.8
        alphas[alpha_mask_low] = 0.4  
        alphas[alpha_mask_med_low] = 0.15  
        color_seg[:,:,3] = alphas  

        # Convert to pixel vals out of 255 and save image and base as PIL
        img = Image.fromarray(np.array(color_seg * 255).astype(np.uint8))
        base = image.convert("RGBA")
        
        # Blend together
        blended_img = Image.alpha_composite(base,img)
        blended_img = blended_img.convert("RGB")                       

        return blended_img
    

    # Function to apply trashbot to photo
    def trashbot(image):    

        # Apply model outputs to image
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits 

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )

        # Get probabilities from logits
        trash_pixel_probs = sigmoid(upsampled_logits[0,1,:,:])

        # Apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pred_img = get_seg_overlay(image, pred_seg, trash_pixel_probs)

        # Get the number of trash pixels
        trash_pixels = np.sum(pred_seg.numpy())
        prob_estimate_trash_pixels = np.sum(trash_pixel_probs)

        # Return image with trashbot shading
        return pred_img, trash_pixels, prob_estimate_trash_pixels

    # Loop through ds and do trashbot_predict, save image
    total_trash_pixels = 0
    prob_estimate_trash_pixels = 0
    total_pixels = len(ds['train']) * (5472 * 3648)
    for i in tqdm (range (len(ds['train'])), desc="Trashbot processing images") :

        # Get path to root
        cwd = Path(os.getcwd())
        session_path = str(cwd.parent.absolute()) + "/drone_photos/" + dataset_name          
        
        # Get image and trashbot_predict    
        image = ds['train'][i]['image']
        trashbot_result, trash_pixels, prob_est_trash_pix = trashbot(image)
        total_trash_pixels += trash_pixels
        prob_estimate_trash_pixels += prob_est_trash_pix

        # load original image and extract EXIF
        fpath = list(ds['train'].info.download_checksums.keys())[i]
        original_image = Image.open(fpath)
        exif = original_image.info['exif']

        # Convert to image, apply Gaussian blur, and save with EXIF from original photo
        # trashbot_image = Image.fromarray(trashbot_result)  
        trashbot_result.save(f"{session_path}/Trashbot_v2_no_blur/trashbot_v2_{dataset_name}_{i}.JPG", exif=exif)                
        trashbot_image = trashbot_result.filter(ImageFilter.GaussianBlur(radius=blur_factor))  
        trashbot_image.save(f"{session_path}/Trashbot_v2/trashbot_v2_{dataset_name}_{i}.JPG", exif=exif)                
    
    # Print trash relative to non-trash
    print(f"Total trash pixels: {total_trash_pixels} -- {round((100 * total_trash_pixels / total_pixels),2)}% trash")
    print(f"Total trash pixels soft classification: {prob_estimate_trash_pixels} -- {round((100 * prob_estimate_trash_pixels / total_pixels),2)}% trash")
        
trashbot_predict("S05_Ebo_Town_Riverine", "trashbot", 3)
# trashbot_predict("S06_Dippa_Kunda", "trashbot", 3)