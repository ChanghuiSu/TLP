from utils import save_png
import argparse
from torch.utils.data import DataLoader
from model import *
from data import *
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/path/to/your/dataset/')
parser.add_argument('--generator', type=str, default="./checkpoints/name_of_your_experiment/G_80.pth",
                    help='Generator checkpoint file')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu_id', type=str, default='0', help='The GPU to be used this time.')
parser.add_argument('--save_path',  default='./output/folder_for_saving_generated_images/',
                    help='Folder for saving generated images.')
parser.add_argument("--num_refinement_blocks", type=int, default=2, help="Num of the refinement blocks")
parser.add_argument("--start_num", type=int, default=870, help="Starting number of training samples")
parser.add_argument("--end_num", type=int, default=1000, help="End number of training samples")
parser.add_argument("--start_slice", type=int, default=27, help="Starting number of slice")
parser.add_argument("--end_slice", type=int, default=127, help="End number of slice")
parser.add_argument("--data_list", type=str, default="./data_list/Brats2021_selected.txt",
                    help="A file recording the filtered case IDs, shuffled. The first 850 are used for training, "
                         "20 for validation, and the last 130 for testing.")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

device = "cuda" if opt.use_gpu else "cpu"
print(f"Using device: {device}")

dataloader = DataLoader(
    LoadData(opt.dataroot, ["t1", "t2", "seg"], opt.data_list,
             start_num=opt.start_num, end_num=opt.end_num, start_slice=opt.start_slice,
             end_slice=opt.end_slice),
    batch_size=1,
    shuffle=False,
)
G = TLP(num_refinement_blocks=opt.num_refinement_blocks, probability_of_losing_labels=1.0, device=device)

G.to(device)
G.load_state_dict(torch.load(opt.generator), strict=True)
G.eval()

if not os.path.isdir(opt.save_path):
    os.makedirs(opt.save_path)

for ii, batch in enumerate(dataloader):

    print(str(ii) + "/" + str(len(dataloader)))

    t1_data = batch['t1'][0]
    t2_data = batch['t2'][0]
    edema_region = batch['seg'][0]
    tumor_region = batch['seg'][1]
    t1_path = batch['t1'][1][0]

    t1_data = t1_data.to(device)
    t2_data = t2_data.to(device)
    edema_region = edema_region.to(device)
    tumor_region = tumor_region.to(device)

    pred_out = G(t1_data, t2_data, edema_region, tumor_region)
    save_png(pred_out, opt.save_path, t1_path)
