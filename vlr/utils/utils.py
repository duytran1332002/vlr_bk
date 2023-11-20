import os

import torch
import glob

def save_model_from_ckpt(ckpt_path):
    #check path
    if not os.path.exists(ckpt_path):
        raise Exception("Path does not exist")

    # load checkpoint
    states = torch.load(ckpt_path, map_location=lambda storage, loc: storage)["state_dict"]
    states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
    # get model path
    model_path = os.path.join(os.path.dirname(ckpt_path), "model.pth")
    print(model_path)
    torch.save(states, model_path)

if __name__ == "__main__":
    path = '/home/duytran/Desktop/vlr/last.ckpt'
    save_model_from_ckpt(path)