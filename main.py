import torch
import cv2 as cv
from OneFrameProcessing import loader, image_loader, calculate_delta_e,color_transfer_pipeline,algorithmic_models_generation,calculate_psnr, calculate_ssim,split_video_into_frames

import torchvision.models as models
import os
import time
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip
from PIL import Image, ImageChops


if __name__ == "__main__":

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.set_default_device(device)
    imsize = 512 if torch.accelerator.is_available() else 128

    vgg19 = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    while True:

        choice = int(input('What type of file would you like to process? (1 for images, 2 for videos, 3 for benchmarks, 0 for exit)\n'))

        if choice == 1:    
            source_image = input('\nProvide the name of the source image:\n')
            style_image = input('\nProvide the name of the reference image:\n')
            print('\n')
            style_image_path = f"References/{style_image}"
            source_image_path = f"OneFrameOriginals/{source_image}"
            intermediate_image_path = f"OneFrameIntermediate/{style_image[:-4]}_{source_image[:-4]}_Intermediate.jpg"
            final_image_path = f"OneFrameFinal/{style_image[:-4]}_{source_image[:-4]}_Final.jpg"
            

            style_image_tensor = image_loader(style_image_path)
            source_image_tensor = image_loader(source_image_path)


            intermediate_result = color_transfer_pipeline(vgg19, vgg_mean, vgg_std, style_image_path ,source_image_path,intermediate_image_path)
            
        
            img_cv = cv.imread(intermediate_image_path)
            blur_cv = cv.bilateralFilter(img_cv, 9,75,75)
            blur_rgb = cv.cvtColor(blur_cv, cv.COLOR_BGR2RGB)
            cv.imwrite( f"OneFrameIntermediate/{style_image[:-4]}_{source_image[:-4]}_Intermediate_Blur.jpg", blur_cv) 

            image1 = Image.fromarray(blur_rgb)
            image2 = Image.open(source_image_path).convert("RGB").resize(image1.size)

            final_result = ImageChops.hard_light(image1, image2)
            algorithmic_models_generation(style_image,source_image)
            final_result.save(final_image_path)
            
            final_image_tensor = loader(final_result).unsqueeze(0)
            final_image_tensor.to(device, torch.float)

            vgg_cdm_delta = calculate_delta_e(final_image_tensor,style_image_tensor)

            print(f'VGG-CDM Delta-E:              {vgg_cdm_delta:.4f}')
            print(f"\nOutput Intermediate: {intermediate_image_path}")
            print(f'Output Algorithmic Models: AlgorithmicModelsOutput/...')
            print(f"Output Final: {final_image_path}\n")
        
        elif choice == 2:
            source_video = input('\nProvide the name of the source video:\n')
            style_image = input('\nProvide the name of the reference image:\n')

            split_video_into_frames(f'Videos/{source_video}',style_image)

            style_image_path = f"References/{style_image}"
            source_video_path = f"Videos/{source_video}"

            
            DIR = f'Videos/{source_video[:-4]}_{style_image[:-4]}_Frames'
            frameNum = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            
            vgg19 = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
            vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            style_tensor = image_loader(style_image_path)
            try:
                os.mkdir(f'Videos/{source_video[:-4]}_{style_image[:-4]}_FramesIntermediate')
            except:
                print("The directory already exists.")

            try:
                os.mkdir(f'Videos/{source_video[:-4]}_{style_image[:-4]}_FramesFinal')
            except:
                print("The directory already exists.")

            for i in range(frameNum):
                start_time = time.time()
                print(f'\nFrame: {i+1}')
                original_frame_path = f"{source_video_path[:-4]}_{style_image[:-4]}_Frames/Frame_{i}.jpg"
                intermediate_frame_path = f"{source_video_path[:-4]}_{style_image[:-4]}_FramesIntermediate/IntermediateFrame_{i}.jpg"
                final_frame_path  = f"{source_video_path[:-4]}_{style_image[:-4]}_FramesFinal/FinalFrame_{i}.jpg"
            
                intermediate_result = color_transfer_pipeline(vgg19, vgg_mean, vgg_std, style_image_path ,original_frame_path,intermediate_frame_path)
                
                

                img_cv = cv.imread(intermediate_frame_path)
                blur_cv = cv.bilateralFilter(img_cv, 9,75,75)
                blur_rgb = cv.cvtColor(blur_cv, cv.COLOR_BGR2RGB)
        

                image1 = Image.fromarray(blur_rgb)
                image1.save(f'Videos/{source_video[:-4]}_{style_image[:-4]}_FramesIntermediate/IntermediateFrame{i}_Blur.jpg') 
                image2 = Image.open(original_frame_path).convert("RGB").resize(image1.size)

                final_result = ImageChops.hard_light(image1, image2)

                final_result.save(final_frame_path)
                
                final_image_tensor = loader(final_result).unsqueeze(0)
                final_image_tensor.to(device, torch.float)

                vgg_cdm_delta = calculate_delta_e(final_image_tensor,style_tensor)
                print(f'VGG-CDM Delta-E:              {vgg_cdm_delta:.4f}')
                print(f"Output Intermediate:          {intermediate_frame_path}")
                print(f"Output Final:                 {final_frame_path}\n")
                print(f'This frame took: {(time.time()-start_time):.4f} seconds to process')
                print(90 * '-')
                
            output_path = f'Videos/VideosFinal/{source_video[:-4]}_{style_image[:-4]}.mp4'
            
            image_files = []

            for j in range(frameNum): 
            
                image_files.append(f"Videos/{source_video[:-4]}_{style_image[:-4]}_FramesFinal/FinalFrame_{j}.jpg") 

            cam = cv.VideoCapture(source_video_path)
            fps = cam.get(cv.CAP_PROP_FPS)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(output_path)


            source_clip = VideoFileClip(source_video_path)
            target_clip = VideoFileClip(output_path)
            extracted_audio = source_clip.audio
            extracted_audio = extracted_audio.subclip(0, target_clip.duration)
            final_clip = target_clip.set_audio(extracted_audio)

            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        elif choice == 3:
                delta_list_vgg = []
                delta_list_lhm = []
                delta_list_pccm = []
                delta_list_reinhard = []
                delta_list_intermediate = []

                DIR_ORIGINALS = 'OneFrameOriginals'
                DIR_REFERENCES = 'References'
                DIR_ALGO = 'AlgorithmicModelsOutput'
                DIR_FINAL = 'OneFrameFinal'
                count = 0 # keeps track of the number of files processed
                # one_frame_len = len([name for name in os.listdir(DIR_ORIGINALS) if os.path.isfile(os.path.join(DIR_ORIGINALS, name))])
                # references_len = len([name for name in os.listdir(DIR_REFERENCES) if os.path.isfile(os.path.join(DIR_REFERENCES, name))])

                # matches every image from OneFrameOriginals with every Reference
                for source_image in os.scandir(DIR_ORIGINALS):  
                        for style_image in os.scandir(DIR_REFERENCES):

                            try:
                                source_image = source_image.name
                            except:
                                pass
                            try:
                                style_image = style_image.name
                            except:
                                pass
                            print('\n')
                            print(f'Source image: { source_image }')
                            print(f'Style image: {style_image} ')

                            style_image_path = f"References/{style_image}"
                            source_image_path = f"OneFrameOriginals/{source_image}"
                            intermediate_image_path = f"OneFrameIntermediate/{style_image[:-4]}_{source_image[:-4]}_Intermediate.jpg"
                            final_image_path = f"OneFrameFinal/{style_image[:-4]}_{source_image[:-4]}_Final.jpg"
                            style_image_tensor = image_loader(style_image_path)
                    
                            intermediate_result = color_transfer_pipeline(vgg19, vgg_mean, vgg_std, style_image_path ,source_image_path,intermediate_image_path)
                            
                            intermediate_image_tensor = image_loader(intermediate_image_path)
                            final_delta_intermediate = calculate_delta_e(intermediate_image_tensor, style_image_tensor)
                            delta_list_intermediate.append(final_delta_intermediate)
                            
                            img_cv = cv.imread(intermediate_image_path)
                            blur_cv = cv.bilateralFilter(img_cv, 9,75,75)
                            blur_rgb = cv.cvtColor(blur_cv, cv.COLOR_BGR2RGB)
                            cv.imwrite( f"OneFrameIntermediate/{style_image[:-4]}_{source_image[:-4]}_Intermediate_Blur.jpg", blur_cv) 

                            image1 = Image.fromarray(blur_rgb)
                            image2 = Image.open(source_image_path).convert("RGB").resize(image1.size)

                            final_result = ImageChops.hard_light(image1, image2)
                            final_result.save(final_image_path)
                            
                            source_image_tensor = image_loader(source_image_path)
                            final_image_tensor = image_loader(final_image_path)
                            final_delta_vgg = calculate_delta_e(final_image_tensor, style_image_tensor)

                            delta_list_vgg.append(final_delta_vgg)
                            count += 1
                            print(f"VGG-CDM Delta E:              {final_delta_vgg:.4f}") 

                            algorithmic_models_generation(style_image,source_image)
                            lhm_image_tensor = image_loader(f'AlgorithmicModelsOutput/{style_image[:-4]}_{source_image[:-4]}_output_lhm.jpg')
                            pccm_image_tensor = image_loader(f'AlgorithmicModelsOutput/{style_image[:-4]}_{source_image[:-4]}_output_pccm.jpg')
                            reinhard_image_tensor = image_loader(f'AlgorithmicModelsOutput/{style_image[:-4]}_{source_image[:-4]}_output_reinhard.jpg')

                            final_delta_lhm = calculate_delta_e(lhm_image_tensor, style_image_tensor)
                            final_delta_pccm = calculate_delta_e(pccm_image_tensor, style_image_tensor)
                            final_delta_reinhard = calculate_delta_e(reinhard_image_tensor, style_image_tensor)
                
                            print(f'LHM Delta E:                  {final_delta_lhm:.4f}')
                            print(f'PCCM Delta E:                 {final_delta_pccm:.4f}')
                            print(f'Reinhard Delta E:             {final_delta_reinhard:.4f}\n')
                            delta_list_lhm.append(final_delta_lhm)
                            delta_list_pccm.append(final_delta_pccm)
                            delta_list_reinhard.append(final_delta_reinhard)


                            print(f"Output Intermediate: {intermediate_image_path}")
                            print(f"Output Final: {final_image_path}")
                            print(f'Output Algorithmic Models: AlgorithmicModelsOutput/...')
                            print(90 * '-')
                average_delta_vgg_intermediate = sum(delta_list_intermediate) / count
                average_delta_vgg = sum(delta_list_vgg) / count
                average_delta_lhm = sum(delta_list_lhm) / count
                average_delta_pccm = sum(delta_list_pccm) / count
                average_delta_reinhard = sum(delta_list_reinhard) / count

                # PSNR Calculation
                psnr_lhm = 0
                psnr_pccm = 0
                psnr_reinhard = 0
                psnr_vgg = 0
                psnr_intermediate = 0
                ssim_lhm = 0
                ssim_pccm = 0
                ssim_reinhard = 0
                ssim_vgg = 0
                ssim_intermediate = 0
                image_amount = len([name for name in os.listdir(DIR_FINAL) if os.path.isfile(os.path.join(DIR_FINAL, name))])

                # print(f'Algo models output: {len([name for name in os.listdir('AlgorithmicModelsOutput') if os.path.isfile(os.path.join('AlgorithmicModelsOutput', name))])}')
                # print(f'One frame final: {len([name for name in os.listdir('OneFrameFinal') if os.path.isfile(os.path.join('OneFrameFinal', name))])}')
                # print(f'One frame intermediate: {len([name for name in os.listdir('OneFrameIntermediate') if os.path.isfile(os.path.join('OneFrameIntermediate', name))])}')
                algo_count = 0
                oneframe_count = 0
                oneframe_inter = 0
                for source_image in os.scandir('OneFrameOriginals'):
                    source_image = source_image.name
                    for processed in os.scandir('AlgorithmicModelsOutput'):
                        processed = processed.name
                        if source_image[:-4] in processed[:-4] and 'lhm' in processed:
                            psnr_lhm += calculate_psnr(f'OneFrameOriginals/{source_image}',f'AlgorithmicModelsOutput/{processed}')
                        elif source_image[:-4] in processed[:-4] and 'pccm' in processed:
                            psnr_pccm += calculate_psnr(f'OneFrameOriginals/{source_image}',f'AlgorithmicModelsOutput/{processed}')
                        elif source_image[:-4] in processed[:-4] and 'reinhard' in processed:
                            psnr_reinhard += calculate_psnr(f'OneFrameOriginals/{source_image}',f'AlgorithmicModelsOutput/{processed}')
                        algo_count +=1
                

                for source_image in os.scandir('OneFrameOriginals'):
                    source_image = source_image.name
                    for processed in os.scandir('OneFrameFinal'):
                        processed = processed.name
                        if source_image[:-4] in processed[:-4]:
                            psnr_vgg += calculate_psnr(f'OneFrameOriginals/{source_image}',f'OneFrameFinal/{processed}')
                            oneframe_count+=1
                    for processed2 in os.scandir('OneFrameIntermediate'):
                        processed2 = processed2.name
                        if source_image[:-4] in processed2[:-4] and 'Blur' not in processed2[:-4]:
                            psnr_intermediate += calculate_psnr(f'OneFrameOriginals/{source_image}',f'OneFrameIntermediate/{processed2}')
                            oneframe_inter+=1

                
                #SSIM Calculations

                for source_image in os.scandir('OneFrameOriginals'):
                    source_image = source_image.name
                    for processed in os.scandir('AlgorithmicModelsOutput'):
                        processed = processed.name
                        if source_image[:-4] in processed[:-4] and 'lhm' in processed:
                            ssim_lhm += calculate_ssim (f'OneFrameOriginals/{source_image}',f'AlgorithmicModelsOutput/{processed}')
                        elif source_image[:-4] in processed[:-4] and 'pccm' in processed:
                            ssim_pccm += calculate_ssim (f'OneFrameOriginals/{source_image}',f'AlgorithmicModelsOutput/{processed}')
                        elif source_image[:-4] in processed[:-4] and 'reinhard' in processed:
                            ssim_reinhard += calculate_ssim (f'OneFrameOriginals/{source_image}',f'AlgorithmicModelsOutput/{processed}')

                for source_image in os.scandir('OneFrameOriginals'):
                    source_image = source_image.name
                    for processed in os.scandir('OneFrameFinal'):
                        processed = processed.name
                        if source_image[:-4] in processed[:-4]:
                            ssim_vgg += calculate_ssim (f'OneFrameOriginals/{source_image}',f'OneFrameFinal/{processed}')
                    for processed2 in os.scandir('OneFrameIntermediate'):
                        processed2 = processed2.name
                        if source_image[:-4] in processed2[:-4] and 'Blur' not in processed2[:-4]:
                            ssim_intermediate += calculate_ssim (f'OneFrameOriginals/{source_image}',f'OneFrameIntermediate/{processed2}')
                print(f"One frame Count: {oneframe_count}, One Frame Inter: {oneframe_inter}, AlgoModels: {algo_count}")             
                #Final Results
                print(image_amount)
                print(f'\nSSIM LHM:                                     {ssim_lhm/(image_amount):.4f}')
                print(f'SSIM PCCM:                                    {ssim_pccm/(image_amount):.4f}')
                print(f'SSIM Reinhard:                                {ssim_reinhard/(image_amount):.4f}')
                print(f'SSIM VGG-CDM:                                 {ssim_vgg/(image_amount):.4f}')
                print(f'SSIM Intermediate VGG-CDM:                    {ssim_intermediate/(image_amount):.4f}')

                print(f'\nPSNR LHM:                                     {psnr_lhm/(image_amount):.4f}')
                print(f'PSNR PCCM:                                    {psnr_pccm/(image_amount):.4f}')
                print(f'PSNR Reinhard:                                {psnr_reinhard/(image_amount):.4f}')
                print(f'PSNR VGG-CDM:                                 {psnr_vgg/(image_amount):.4f}')
                print(f'PSNR Intermediate VGG-CDM:                    {psnr_intermediate/(image_amount):.4f}')


                print(f'\nFinal Average Delta E VGG-19 Intermediatej:   {average_delta_vgg_intermediate:.4f}')
                print(f'Final Average Delta E VGG-19:                 {average_delta_vgg:.4f}')
                print(f'Final Avergae Delta E LHM Algorithm:          {average_delta_lhm:.4f}')
                print(f'Final Avergae Delta E PCCM Algorithm:         {average_delta_pccm:.4f}')
                print(f'Final Avergae Delta E Reinhard Algorithm:     {average_delta_reinhard:.4f}') 
            
        elif choice == 0:
                break