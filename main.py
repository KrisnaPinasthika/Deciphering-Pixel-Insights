import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from diodetools.DiodeLoader import DiodeDataLoader
from diodetools.TrainTest import train, load_state 
from Model.MonoAGIResNet import MonoAGIResNet
from Model.MonoAGIDenseNet import MonoAGIDenseNet
from Model.MonoAGIEfficientV2 import MonoAGIEfficientV2

torch.random.manual_seed(1)
np.random.seed(1)

def getData(path):
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    TRAIN_PATH = r"../dataset/diode_indoor/train"
    TEST_PATH = r'../dataset/diode_indoor/test'
    
    df_train = getData(TRAIN_PATH)

    height, width = 192, 256
    max_depth = 50
    trainloader = DataLoader(
                        DiodeDataLoader(
                            data_frame=df_train, 
                            max_depth=max_depth, 
                            img_dim=(height, width), 
                            depth_dim=(height, width)
                            ),
                        batch_size=8,
                        shuffle=True, 
                        num_workers=4, 
                        pin_memory=True
                    )

    device = torch.device("cuda")
    backbone = 'efficient_v2_s'
    model_name = f'{backbone}'
    model = MonoAGIEfficientV2(device=device, backbone_name=backbone, freeze=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_weight = .1
    epochs = 10
    
    train(
        model=model, 
        model_name=model_name, 
        max_depth=max_depth,
        l1_weight=l1_weight, 
        loader=trainloader, 
        epochs=epochs,
        optimizer=optimizer, 
        device=device, 
        save_model=False, 
        save_train_state=False
    )

    torch.cuda.empty_cache()