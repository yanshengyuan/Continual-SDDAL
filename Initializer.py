import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import sys

from Official_batched_fitting_DIC_fullscene.utils.loader import get_training_data

from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import CircularAperture

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('-b', '--init_size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--beamshape', default='rec', type=str)
parser.add_argument('--caustic_plane', default='prefoc', type=str)
parser.add_argument('--vis_path', default='', type=str)

if __name__ == '__main__':
    
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    
    vis_dir = args.vis_path
    beamshape = args.beamshape
    plane = args.caustic_plane
    size = args.init_size
    
    lightsource_path = 'M290_MachineSimu_GPU/lightsource_full_scene.npy'
    
    '''
    #-------------------------Sanity check with RecTophat starts---------------------------
    
    from M290_MachineSimu_GPU.M290_sanity_check import M290
    machine = M290(1 ,args.beamshape, lightsource_path, device)
    machine = machine.to(device)
    near_field = machine.nearField
    
    batchsize = 1
    train_dataset = get_training_data('Official_batched_fitting_DIC_fullscene')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)
    
    print("Sanity check sample generation with RecTophat starts now!")
    for i, (beamshape, Z, name) in enumerate(train_loader):
        
        print(name)
        
        beamshape = beamshape.to(device)
        Zernike_Coeffs = Z.to(device, non_blocking=True).squeeze()
        with torch.no_grad():
            machine.zernike_coeffs.copy_(Zernike_Coeffs.view(batchsize, 12, 1, 1))
        
        imaging_field, phase = machine(near_field)
        I = torch.abs(imaging_field)**2
        max_per_sample = I.amax(dim=(1,2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.detach().cpu().numpy()
        
        phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
        phase = phase[: , machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
        phase = phase.detach().cpu().numpy()
        
        for j in range(len(I)):
            np.save("rec_sanity_check/"+name[j][:7]+'_intensity.npy', I[j])
            np.save("rec_sanity_check/"+name[j][:7]+'_phase.npy', phase[j])
    print("Sanity check sample generation with RecTophat has ended!")
    
    #-------------------------Sanity check with RecTophat ends---------------------------
    '''
    
    from M290_MachineSimu_GPU.M290_full_scene import M290
    machine = M290(1 ,args.beamshape, lightsource_path, device)
    machine = machine.to(device)
    near_field = machine.nearField
    
    for j in range(size):
        # Random initialization
        print("Initial training sample " + str(j))
        with torch.no_grad():
            torch.nn.init.uniform_(machine.zernike_coeffs, a=-1.5, b=1.5)
            
        imaging_field, phase = machine(near_field)
        I = torch.abs(imaging_field)**2
        max_per_sample = I.amax(dim=(1,2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.detach().cpu().numpy().squeeze()
        
        phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
        phase = phase[: , machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
        phase = phase.detach().cpu().numpy().squeeze()
        
        z = machine.zernike_coeffs.squeeze().cpu().detach().numpy().squeeze()
        
        mpimg.imsave(vis_dir+'/training_set/intensity/img/'+'intensity_0_' + str(j) +'.png', -I, cmap='Greys')
        np.save(vis_dir+'/training_set/intensity/npy/'+'intensity_0_' + str(j) +'.npy', I)
        mpimg.imsave(vis_dir+'/training_set/phase/img/'+'phase_0_' + str(j) +'.png', phase, cmap='Greys')
        np.save(vis_dir+'/training_set/phase/npy/'+'phase_0_' + str(j) +'.npy', phase)
        np.save(vis_dir+'/training_set/zernikes/'+'zernikes_0_' + str(j) +'.npy', z)
        
    #coeffs_np = machine.zernike_coeffs.detach().cpu().numpy()
    #np.save(vis_dir+'/training_set/zernikes_init.npy', coeffs_np)