import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
import json
import math
import os
import csv
from tqdm import tqdm
import pickle
from VideoMAEv2.models.modeling_finetune import vit_base_patch16_224
# from VideoMAE.modeling_finetune import vit_base_patch16_224
##videomae, videomaev2는 github에서 다운받아서 사용
# videomae : https://github.com/MCG-NJU/VideoMAE
# videomaev2 : https://github.com/OpenGVLab/VideoMAEv2
# trained model은 model zoo에 있다.

#add by jamwon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=False),
    ##model마다 normalize방법이 다르기 때문에 확인할 것!
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #videomae  v1, v2
    # transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]) #slowfast
    # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) #clip
])

def video2numpy(path, size = 224, sr=1):
    #video를 읽어와서 numpy로 변환

    vr = VideoReader(path, width=size, height=size) #HWC
    fps = int(round((vr.get_avg_fps())))
    frame_id_list = range(0, len(vr), sr)
    frames = vr.get_batch(frame_id_list).asnumpy()

    return frames, fps

def get_model(model_name = 'slowfast') :
    #원하는 model 있으면 추가해서 사용
    num_frame = 16
    kinetics = 400

    if model_name == 'slowfast' :
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        num_frame = 32
        # model.blocks[6] = nn.Identity()

    elif model_name == 'videomaev2' :
        model = vit_base_patch16_224(num_classes = 710)
        #checkpoint = torch.load('VideoMAEv2/vit_b_k710_dl_from_giant.pth', map_location="cpu")
        checkpoint = torch.load('VideoMAEv2/vit_b_k710_dl_from_giant.pth', map_location=device)

        kinetics = 710
        model.load_state_dict(checkpoint["module"])
        # model.head = nn.Identity()


    elif model_name == 'videomae' :
        model = vit_base_patch16_224(num_classes = 400)
        checkpoint = torch.load('VideoMAE/checkpoint.pth', map_location=device)
        kinetics = 400
        model.load_state_dict(checkpoint["module"])
        # model.head = nn.Identity()
    else :
        exit()

    model.eval()
    #model.cpu()
    model.to(device)

    return model, num_frame, kinetics

def extract_video(frames, fps, num_frames, save_path, model, model_name, kinetics_class) :
    sec = 2

    sampling_term = sec * fps
    seq_len = len(frames)
    sampling_list = [i for i in range(0, seq_len, sampling_term)]
    reminder = sampling_term - seq_len % sampling_term
    print("ev0")
    video = []
    i = 0
    print(len(frames))
    for frame in frames :
        print(i)
        i+=1
        video.append(transform(frame))
    print("ev1")
    
    for _ in range(reminder) :
        video.append(torch.zeros(3, 224, 224))
    print("ev2")

    #video = torch.stack(video, dim = 1).unsqueeze(0)
    video = torch.stack(video, dim = 1).unsqueeze(0).to(device)

    print("ev3")

    feat = []
    #idx = torch.linspace(0, sampling_term - 1, num_frames).long()
    idx = torch.linspace(0, sampling_term - 1, num_frames).long().to(device)
    
    print("ev4")

    with torch.no_grad() :

        for i in sampling_list :
            s = idx + i
            inputs = torch.index_select(video, 2, s)
            if model_name == 'slowfast' :
                fast = inputs.cpu()
                slow = fast[:, :, ::4].cpu()
                output = model([slow, fast]).cpu().squeeze()
                feat.append(output)

            elif model_name == 'videomaev2' :
                #inputs = inputs.cpu()
                #output = model(inputs).cpu().squeeze()
                output = model(inputs).squeeze()
                feat.append(output)

            elif model_name == 'videomae' :
                #inputs = inputs.cpu()
                #output = model(inputs).cpu().squeeze()
                output = model(inputs).squeeze()
                feat.append(output)
        print("ev5")
        output = torch.stack(feat, dim = 0)
        print("ev6")
        preds = F.softmax(output, dim = -1)
        print("ev7")
        value, indices = preds.topk(k=5)
        print("ev8")


        d = {}
        for k, index in enumerate(indices) :
            pred_class_names = [kinetics_class[int(i)] for i in index]
            d[k] = [pred_class_names, value[k]]
        print("ev9")
        # key : n번째 shot, value : class - score
        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump(d, f)
        print("ev10")


parser = argparse.ArgumentParser()
parser.add_argument("--feature_extractor", type=str, default='videomaev2')

args = vars(parser.parse_args())
feat_name = args['feature_extractor']

#dir = '../../TVSum/videos/'
dir = './videos/'

print("jaewon-3")
model, num_frame, kinetics = get_model(feat_name)
print("model,num_frame,kinetics = ",model,num_frame,kinetics)
print("jaewon-2")

kinetics_class = []
print("kinetics_class = ",kinetics_class)
print("jaewon-1")
label_name = F'label_map_k{kinetics}.txt' ## class에 맞게 txt 변경
print("label_name= ",label_name)
with open(label_name, 'r') as f :
    print("jaewon0")
    kinetics_class = f.read().splitlines()
    print(kinetics_class)

for i in tqdm(range(1,2)) :
    video_key = f'video_{i}'
    tqdm.write(f'extract {video_key}.mp4')
    video_path = dir + video_key +'.mp4'
    save_path = video_key + '_output'
    print("now = ",i)
    print("jaewon1")
    frames, fps = video2numpy(video_path)
    print("frames, fps = ",frames, fps)
    print("jaewon2")
    feat = extract_video(frames, fps, num_frame, save_path, model, feat_name, kinetics_class)
    print("feat = ",feat)
    print("jaewon3")
    print("now = ",i)



