import os
import numpy as np
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import torch

batch_size = 1
_gpu_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.frcm_gpu')
try:
    with open(_gpu_file) as _f:
        _gpu_id = int(_f.read().strip())
except Exception:
    _gpu_id = 0
device = torch.device(f"cuda:{_gpu_id}" if torch.cuda.is_available() else "cpu")


def calculate_frcm(img1, img2):
    device = img1.device
    nz, nx, ny = img1.shape
    rnyquist = nx // 2

    half = nx // 2
    x = torch.cat((torch.arange(0, half), torch.arange(-half, 0))).to(device)
    y = x

    X, Y = torch.meshgrid(x, y, indexing='ij')
    r_map = X ** 2 + Y ** 2
    index = torch.round(torch.sqrt(r_map.float()))

    r = torch.arange(0, rnyquist + 1, device=device).float()

    F1 = torch.fft.fft2(img1).permute(1, 2, 0)
    F2 = torch.fft.fft2(img2).permute(1, 2, 0)

    C_r = torch.empty(rnyquist + 1, batch_size, device=device)
    C_i = torch.empty_like(C_r)
    C1  = torch.empty_like(C_r)
    C2  = torch.empty_like(C_r)

    for ii in r:
        auxF1 = F1[torch.where(index == ii)]
        auxF2 = F2[torch.where(index == ii)]
        ii = ii.int()

        real1, imag1 = auxF1.real, auxF1.imag
        real2, imag2 = auxF2.real, auxF2.imag

        C_r[ii] = torch.sum(real1 * real2 + imag1 * imag2, dim=0)
        C_i[ii] = torch.sum(imag1 * real2 - real1 * imag2, dim=0)
        C1[ii]  = torch.sum(real1 ** 2 + imag1 ** 2, dim=0)
        C2[ii]  = torch.sum(real2 ** 2 + imag2 ** 2, dim=0)

    FRC  = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
    FRCm = 1 - torch.where(FRC != FRC, torch.tensor(1.0, device=device), FRC)
    My_FRCloss = torch.mean(FRCm ** 2)
    return My_FRCloss


gt_folder = './Phi_gt/npy'
pred_folder = './Phi_pred/npy'
mae_list = []
ssim_list = []
frcm_list = []

cnt=0
# Traverse through all files in the gt folder
for gt_filename in os.listdir(gt_folder):
    # Load ground truth and prediction arrays
    gt_path = os.path.join(gt_folder, gt_filename)
    pred_path = os.path.join(pred_folder, gt_filename)

    if os.path.exists(pred_path):
        gt_array = np.load(gt_path)
        pred_array = np.load(pred_path)

        diff = np.abs(gt_array - pred_array)
        mae = np.mean(diff)
        mae_list.append(mae)

        ssim_value = ssim(gt_array, pred_array, data_range=gt_array.max() - gt_array.min())
        ssim_list.append(ssim_value)

        gt_tensor   = torch.from_numpy(gt_array.astype(np.float32)).unsqueeze(0).to(device)
        pred_tensor = torch.from_numpy(pred_array.astype(np.float32)).unsqueeze(0).to(device)
        frcm = calculate_frcm(gt_tensor, pred_tensor)
        frcm_list.append(frcm.squeeze().cpu().numpy().item())

        cnt+=1
        print(cnt)

mean_mae  = np.mean(mae_list)
mean_ssim = np.mean(ssim_list)
mean_frcm = np.mean(frcm_list)

print(f"Mean MAE: {mean_mae}")
print(f"Mean SSIM: {mean_ssim}")
print(f"Mean FRCM: {mean_frcm}")