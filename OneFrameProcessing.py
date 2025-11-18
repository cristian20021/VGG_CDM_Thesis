import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageChops
import numpy as np
import copy
from skimage import color
import colortrans
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import os

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device)

# set the desired output
imsize = 512 if torch.accelerator.is_available() else 128

# resizes and transforms the image into a tensor
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

# load image and preprocess it
def image_loader(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# save the tensor as a image
def save_image(tensor, path):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(path, quality=100)


# computes delta e (CIEDE2000) between two images
def calculate_delta_e(img1_tensor, img2_tensor):

    # convert to numpy
    rgb1_np = img1_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    rgb2_np = img2_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    
    # convert to LAB
    lab1_np = color.rgb2lab(np.clip(rgb1_np, 0, 1))
    lab2_np = color.rgb2lab(np.clip(rgb2_np, 0, 1))
    
    # calculate CIEDE2000 Delta E
    delta_e_np = color.deltaE_ciede2000(lab1_np, lab2_np)
    return delta_e_np.mean()

# normalization layer for VGG-19
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
# loss based on color statistics matching
class ColorStatisticsLoss(nn.Module):
    def __init__(self, target_feature):
        # mean and standard deviation of the features of the reference image
        super(ColorStatisticsLoss, self).__init__()
        self.target_mean = torch.mean(target_feature, dim=[2, 3], keepdim=True).detach()
        self.target_std = torch.std(target_feature, dim=[2, 3], keepdim=True).detach()
        self.loss = 0

    def forward(self, input):
        input_mean = torch.mean(input, dim=[2, 3], keepdim=True)
        input_std = torch.std(input, dim=[2, 3], keepdim=True)
        
        mean_loss = nn.functional.mse_loss(input_mean, self.target_mean)
        std_loss = nn.functional.mse_loss(input_std, self.target_std)
        
        self.loss = mean_loss + std_loss
        return input


def build_color_transfer_model(cnn, normalization_mean, normalization_std,
                                style_image, source_image,
                                color_layers=['conv_1']):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    color_losses = []
    model = nn.Sequential(normalization)

    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'

        model.add_module(name, layer)

        if name in color_layers:
            target_feature = model(style_image).detach()
            color_loss = ColorStatisticsLoss(target_feature)
            model.add_module(f"color_loss_{i}", color_loss)
            color_losses.append(color_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ColorStatisticsLoss):
            break
    model = model[:(i + 1)]

    return model, color_losses


def color_transfer_training_loop (cnn, normalization_mean, normalization_std,
                                 style_image, source_image, num_steps=200):

    
    
   
    initial_delta_e = calculate_delta_e(source_image, style_image)

    
    # build model
    model, color_losses = build_color_transfer_model(
        cnn, normalization_mean, normalization_std, style_image, source_image)


    input_img = source_image.clone()
    input_img.requires_grad_(True)
    

    optimizer = optim.Adam([input_img], lr=0.01)
    
    best_delta_e = float('inf')
    best_img = input_img.clone()
    
    for step in range(num_steps):
        optimizer.zero_grad()
        # forward pass through VGG
        model(input_img)
        
        #  color matching loss
        color_loss = sum(cl.loss for cl in color_losses)

        content_loss = nn.functional.mse_loss(input_img, source_image)
        
        total_loss = color_loss + 0.3 * content_loss
        
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_([input_img], max_norm=1.0)
        
        optimizer.step()
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if (step + 1) % 20 == 0 or (step) == 0:
            with torch.no_grad():
                current_delta_e = calculate_delta_e(input_img, style_image)
                if current_delta_e < best_delta_e:
                    best_delta_e = current_delta_e
                    best_img = input_img.clone().detach()
            
            print(f"Step {step+1}: Total Loss: {total_loss.item():.4f}, "
                  f"Delta E: {current_delta_e:.4f}, Best: {best_delta_e:.4f}")
    

    final_delta_e = calculate_delta_e(best_img, style_image)
    

    print(f"\nInitial Delta E:              {initial_delta_e:.4f}")
    print(f"VGG-CDM Intermediate Delta E: {final_delta_e:.4f}")

    
    return best_img
def color_transfer_pipeline(vgg19, vgg_mean, vgg_std,reference_path, target_path, output_path):
   
    style_image = image_loader(reference_path)
    source_image = image_loader(target_path)




    output_img = color_transfer_training_loop (vgg19, vgg_mean, vgg_std, 
                                             style_image, source_image, 
                                             num_steps=200)

    save_image(output_img, output_path)

    
    return output_img

def calculate_psnr( img1, img2, max_value=255 ):

    print( img1 ," , ", img2 )
    img1 = cv.imread( img1 )
    img2 = cv.imread( img2 )
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv.resize( img2, ( img1.shape[1], img1.shape[0] ), interpolation=cv.INTER_AREA)
    psnr = cv.PSNR( img1, img2 )
   
    print(f'PSNR: { psnr }')

    return psnr

def calculate_ssim ( img1, img2 ) :
    print( img1 ," , ", img2 )

    img1 = cv.imread( img1, 0 )
    img2 = cv.imread( img2, 0 )
    dim = ( 512,512 )
    img1 = cv.resize( img1, dim )
    img2 = cv.resize( img2, dim )

    ssim_score, dif = ssim( img1, img2, full=True )

    print(f'SSIM: { ssim_score }')
    return ssim_score
    


def algorithmic_models_generation(style_image,source_image):

    with Image.open(f'OneFrameOriginals/{source_image}') as img:
        content = np.array(img.convert('RGB'))
    with Image.open(f'References/{style_image}') as img:
        reference = np.array(img.convert('RGB'))

    output_lhm = colortrans.transfer_lhm(content, reference)
    output_pccm = colortrans.transfer_pccm(content, reference)
    output_reinhard = colortrans.transfer_reinhard(content, reference)

    Image.fromarray(output_lhm).save(f'AlgorithmicModelsOutput/{style_image[:-4]}_{source_image[:-4]}_output_lhm.jpg')
    Image.fromarray(output_pccm).save(f'AlgorithmicModelsOutput/{style_image[:-4]}_{source_image[:-4]}_output_pccm.jpg')
    Image.fromarray(output_reinhard).save(f'AlgorithmicModelsOutput/{style_image[:-4]}_{source_image[:-4]}_output_reinhard.jpg')
    

def split_video_into_frames(source_video,style_image):

    frame_nr = 0
    path = f'{source_video[:-4]}_{style_image[:-4]}_Frames'
    video_capture = cv.VideoCapture(source_video)

    try:
        os.mkdir(path)
    except:
        print("The directory already exists.")

    while (True):
        success, frame = video_capture.read()
    
        if success:
            cv.imwrite(f'{path}/Frame_{frame_nr}.jpg', frame)
        else:
            break
        frame_nr = frame_nr+1

    video_capture.release()
    print('\nThe video was split into frames.')
    print(f'Output: {path}')