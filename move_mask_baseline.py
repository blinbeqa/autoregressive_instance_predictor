import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy import ndimage
import imgaug
import imgaug.augmenters as iaa


def move_masks_lin_reg(inputs, batch_size, input_frames, future_frames, number_of_instances):
    # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
    output = np.zeros((batch_size, future_frames, number_of_instances, 256, 512))
    for b in range(batch_size):
        for n in range(number_of_instances):
            x = np.zeros(input_frames)
            y = np.zeros(input_frames)
            for i in range(input_frames):
                centre_of_mass=ndimage.measurements.center_of_mass(inputs[b, i, n].cpu().numpy())
                if np.isnan(centre_of_mass[0]):
                    x[i] = 0.0
                    y[i] = 0.0
                else:
                    x[i] = centre_of_mass[0]
                    y[i] = centre_of_mass[1]
            # print("center of mass", centre_of_mass)
            # print("x", x)
            # print("y", y)
            if(x[7]>0.0 and x[6]>0.0):
                counter = 0
                distance = 0
                for d in range(input_frames-1):
                    if x[d]>0.0 and x[d + 1]>0.0:
                        distance += np.abs(x[d] - x[d+1])
                        counter += 1
                distance = distance / counter
                #print("mean distance", distance)
                model = np.polyfit(x, y, 1)
                predict = np.poly1d(model)
                centre_of_mass2=ndimage.measurements.center_of_mass(inputs[b, input_frames-1, n].cpu().numpy())
                for f in range(future_frames):
                    first_non_zero_index = next((index for index,value in enumerate(x) if value != 0), None)
                    if(x[input_frames-1]>x[first_non_zero_index]):
                        to_predict = x[input_frames-1] + (distance*(f+1))
                    else:
                        to_predict = x[input_frames-1] - (distance*(f+1))
                    
                    # seq = iaa.Sequential([
                    # # Apply affine transformations to each image.
                    # # Scale/zoom them, translate/move them, rotate them and shear them.
                    # iaa.Affine(
                    #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # )], random_order=False) # apply augmenters in random order
                    
                    aug = iaa.Affine(translate_px={"x": int(predict(to_predict) - centre_of_mass2[1]), "y": int(to_predict - centre_of_mass2[0])})

                    images_aug = aug(images=inputs[b, input_frames-1, n].cpu().numpy())
                    output[b, f, n] = images_aug

                    # print("to_predict", to_predict)
                    # print("prediction", predict(to_predict))

                    centre_of_mass3=ndimage.measurements.center_of_mass(images_aug)
                    #print("centre_of_mass3", centre_of_mass3)
                        #predict(hours_studied)
                
            
            # for i in range(future_frames):
            #     centre_of_mass[i]=ndimage.measurements.center_of_mass(inputs[b, i, n].cpu().numpy())
            #     print("center of mass", centre_of_mass[i])
    #print("output_shape", output.shape)
    
    return output

def move_masks_oracle(inputs, batch_size, input_frames, future_frames, number_of_instances):
    # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
    output = np.zeros((batch_size, future_frames, number_of_instances, 256, 512))
    for b in range(batch_size):
        for n in range(number_of_instances):
            x = np.zeros(1)
            y = np.zeros(1)
            
            centre_of_mass=ndimage.measurements.center_of_mass(inputs[b, input_frames - 1, n].cpu().numpy())
            if np.isnan(centre_of_mass[0]):
                x[0] = 0.0
                y[0] = 0.0
            else:
                x[0] = centre_of_mass[0]
                y[0] = centre_of_mass[1]

            
            for f in range(future_frames):
                
                centre_of_mass2=ndimage.measurements.center_of_mass(inputs[b, input_frames + f, n].cpu().numpy())
                if not np.isnan(centre_of_mass2[0]) and x[0]>0.0 and not np.isnan(centre_of_mass2[1]) and y[0]>0.0:

                    #print("centre_of_mass2", centre_of_mass2)
                    
                    aug = iaa.Affine(translate_px={"x": int(centre_of_mass2[1] - centre_of_mass[1]), "y": int(centre_of_mass2[0] - centre_of_mass[0])})

                    images_aug = aug(images=inputs[b, input_frames-1, n].cpu().numpy())
                    output[b, f, n] = images_aug

                    #print("to_predict", to_predict)
                    #print("prediction", predict(to_predict))

                    centre_of_mass3=ndimage.measurements.center_of_mass(images_aug)
                    #print("centre_of_mass3", centre_of_mass3)
                        #predict(hours_studied)
                
            
            # for i in range(future_frames):
            #     centre_of_mass[i]=ndimage.measurements.center_of_mass(inputs[b, i, n].cpu().numpy())
            #     print("center of mass", centre_of_mass[i])
    #print("output_shape", output.shape)
    
    return output
