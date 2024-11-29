import random
import torch
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
to_flip = transforms.RandomHorizontalFlip(p=1) 
to_512 = lambda x: x.resize((512, 512), Image.Resampling.LANCZOS)
to_256 = lambda x: x.resize((256, 256), Image.Resampling.LANCZOS)

class FaceDataset(Dataset):
    def __init__(self, cfg):
        self.repeat = 1000
        self.cfg = cfg
        self.img_size = cfg.resolution
        self.nframes = cfg.curr_nframes + cfg.pev_nframes
        self.data_root = cfg.data_root

        self.data_list = glob.glob(self.data_root + '/*.pt')
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path = self.data_list[index]
        audio_emb = torch.load(audio_path).float()
        # print(audio_emb.shape)
    
        vid_path = self.data_list[index].replace('wav_feat', 'face_crop')[:-7] #['file_path']
        all_png_path = sorted(glob.glob(os.path.join(vid_path, '*.png')))
        source_frames = int(len(all_png_path))
        video_len = min(source_frames, len(audio_emb))
        all_frames = list(range(video_len))

        # select a random clip
        rand_idx = random.randint(0, video_len - self.nframes - 1)  # avoid out of bounds
        frame_indices = all_frames[rand_idx: rand_idx + self.nframes]
        key_idx = random.sample(all_frames, 1)[0]

        driving_rgb_crop_lst = [self.convert_to_tensor(to_512(Image.open(all_png_path[i]))) for i in frame_indices]  # [f h w c]
        key_frame = self.convert_to_tensor(to_512(Image.open(all_png_path[key_idx])))

        assert (len(driving_rgb_crop_lst) == self.nframes), f'{len(driving_rgb_crop_lst)}, self.nframes={self.nframes}'
        
        driving_rgb_crop_256x256 = torch.concat(driving_rgb_crop_lst, 0)
        driving_rgb_crop_256x256 = F.interpolate(driving_rgb_crop_256x256, size=(self.img_size, self.img_size), mode='bicubic')

        source_image = F.interpolate(key_frame, size=(self.img_size, self.img_size), mode='bicubic')
        source_image = source_image.squeeze(0)

        audio_emb = audio_emb[rand_idx: rand_idx + self.nframes, :]

        assert int(audio_emb.shape[0]) == int(self.nframes), f'audio_emb.shape: {audio_emb.shape}'

        item = {
            'drving_images': driving_rgb_crop_256x256,
            'source_image': source_image,
            'audio_feat': audio_emb
        }

        return item
    
    def convert_to_tensor(self, image):
        if isinstance(image, list):
            image_tensor = [self.to_tensor(img) for img in image]
            image_tensor = torch.stack(image_tensor)  # all images have to be the same size
        else:
            image_tensor = self.to_tensor(image)

        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor[None]

        return image_tensor