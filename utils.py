import matplotlib.pyplot as plt
import torch
import numpy as np 
import torch.nn.functional as F
import datetime
import cv2
# PyTroch version

SMOOTH = 1e-6

alpha = 3.0
beta = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

fig = plt.figure(figsize=(10,10))
def plot_results(ori_image , result_img , result_exist , label , counter):
    #print("Lane Line existing Probability : ",result_exist)
    original_img=plot_img(ori_image)
    #plot_label(label)
    class_image = iou_result(result_img)
    #convert(class_image)
    save_images(original_img , class_image , counter)


def iou_result(outputs):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(0)
    outputs = outputs.reshape(5 ,288,800)
    outputs = torch.tensor(outputs , dtype=torch.float , device='cpu').unsqueeze(0)
    output_prob = F.log_softmax(outputs , dim=1)
    #take the argmax value in the dim 1
    output_class = torch.argmax(output_prob , dim=1)
    output_class = output_class.squeeze(0)
    return output_class # Or thresholded.mean() if you are interested in average across the batch


def convert(mask):
    mask = mask.numpy()
    height , width = mask.shape
    copy_img = np.zeros((height , width , 3) , dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            idx = mask[i][j]
            label = labels[idx]
            color = label.color
            copy_img[i][j]=color
    plt.title("Segment Mask")
    plt.imshow(copy_img)
    plt.show()

def plot_img(image):
    #unnormallize the image
    image = image.reshape(3, 288,800)
    image = np.transpose(image , (1,2,0))
    #plt.title("Original Image")
    #plt.imshow(image)
    #plt.show()
    return image
    
def plot_label(image):
    #unnormallize the image
    image = image.reshape(288,800)
    plt.title("Original Image")
    plt.imshow(image)
    plt.show()

def save_images(original_img , result_img , index):
    #obtain the none zeros indexes on the result image
    result_img = result_img.numpy()*255
    non_zero = result_img > 0
    original_img = cv2.cvtColor(original_img , cv2.COLOR_BGR2RGB)
    original_img = original_img*std + mean
    original_img = original_img*255
    #original_img[non_zero]=[0,255,0]
    res_path=os.path.join("/content/gdrive/My Drive/CULane/result_img",str(index))
    ori_path=os.path.join("/content/gdrive/My Drive/CULane/original_img",str(index))
    res_path= res_path+str(".jpg")
    ori_path= ori_path+str(".jpg")
    cv2.imwrite(ori_path, original_img)
    cv2.imwrite(res_path, result_img)