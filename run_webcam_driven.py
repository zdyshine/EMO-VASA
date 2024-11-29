
import argparse
import os
import cv2
import torch
import numpy as np
from torch import nn
from glob import glob
from PIL import Image
from dataclasses import dataclass
from torchvision import transforms
from torch.nn import functional as F
from tqdm import trange, tqdm
from torchvision.transforms import ToTensor, ToPILImage
from notebooks.infer import InferenceWrapper
from networks.volumetric_avatar import FaceParsing
from repos.MODNet.src.models.modnet import MODNet
from ibug.face_detection import RetinaFacePredictor
import glob

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
to_flip = transforms.RandomHorizontalFlip(p=1) 
to_512 = lambda x: x.resize((512, 512), Image.Resampling.LANCZOS)
to_256 = lambda x: x.resize((256, 256), Image.Resampling.LANCZOS)


def get_bg(s_img, mdnt = True):
    gt_img_t = to_tensor(s_img)[:3].unsqueeze(dim=0).cuda()
    m = get_mask(gt_img_t) if mdnt else get_mask_fp(gt_img_t)
    kernel_back = np.ones((21, 21), 'uint8')
    mask = (m >= 0.8).float()
    mask = mask[0].permute(1,2,0)
    dilate_mask = cv2.dilate(mask.cpu().numpy(), kernel_back, iterations=2)
    dilate_mask = torch.FloatTensor(dilate_mask).unsqueeze(0).unsqueeze(0).cuda()
    background = lama(gt_img_t.cuda(), dilate_mask.cuda())
    bg_img = to_image(background[0])
    bg = to_tensor(bg_img.resize((512, 512), Image.BICUBIC))
    return bg, bg_img

def get_modnet_mask(img):
    im_transform = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    im = im_transform(img)
    ref_size = 512
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda(), True)

    # resize and save matteget_mask
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

    return matte

@torch.no_grad()
def get_mask(source_img_crop):
    source_img_mask = get_modnet_mask(source_img_crop)
    source_img_mask = source_img_mask
    source_img_mask = source_img_mask.clamp_(max=1, min=0)
    
    return source_img_mask


@torch.no_grad()
def get_mask_fp(source_img_crop):
    face_mask_source, _, _, cloth_s = face_idt.forward(source_img_crop)
    trashhold = 0.6
    face_mask_source = (face_mask_source > trashhold).float()

    source_mask_modnet = get_mask(source_img_crop)

    face_mask_source = (face_mask_source*source_mask_modnet).float()

    return face_mask_source


def connect_img_and_bg(img, bg, mdnt=True):
    pred_img_t = to_tensor(img)[:3].unsqueeze(0).cuda()
    _source_img_mask = get_modnet_mask(pred_img_t) if mdnt else get_mask_fp(pred_img_t)
    mask_sss = torch.where(_source_img_mask>0.3, _source_img_mask, _source_img_mask*0)**8
    out_nn = mask_sss.cpu()*pred_img_t.cpu()+ (1-mask_sss.cpu())*bg.cpu()
    return to_image(out_nn[0])



project_dir = os.path.dirname(os.path.abspath(__file__))
args_overwrite = {'l1_vol_rgb':0}
face_idt = FaceParsing(None, 'cuda', project_dir=project_dir)

lama = torch.jit.load('repos/jit_lama.pt').cuda()

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load('repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'))
modnet.eval()

threshold = 0.8
device = 'cuda'
face_detector = RetinaFacePredictor(threshold=threshold, device=device, model=(RetinaFacePredictor.get_model('mobilenet0.25')))

inferer = InferenceWrapper(experiment_name = 'Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1', model_file_name = '328_model.pth',
                           project_dir = project_dir, folder = 'logs', state_dict = None,
                           args_overwrite=args_overwrite, pose_momentum = 0.1, print_model=False, print_params = True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_image_path', type=str, default='data/IMG_1.png', help='Path to source image')

    args = parser.parse_args()

    source_img = to_512(Image.open(args.source_image_path))
    cv2.imshow('source_image', np.array(source_img)[:, :, :3][:,:,::-1])
    source = inferer.convert_to_tensor(to_512(source_img))[:, :3]
    inferer.forward(source, crop=False, smooth_pose=False, target_theta=True, mix=False, mix_old=False, modnet_mask=False)
    bg, bg_img = get_bg(source_img, mdnt=False)

    cap = cv2.VideoCapture(0)
    #isOpened() 用来判断是否捕获到视频
    if not cap.isOpened():
        print("无法打摄像机")
        exit()
    cap.set(3, 1920)  # width=1920
    cap.set(4, 1080)  # height=1080

    while True:
        # 如果正确读取帧，ret为True，cap.read() 方法从摄像头中读取一帧视频
        ret, frame = cap.read()
        if not ret:
            break

        # curr_d = frame[540:,480:1440,:]
        curr_d = frame[540:,640:1180,:] # 1080, 1920 -> 540, 
        cv2.imshow('drving_image', curr_d)

        curr_d = inferer.convert_to_tensor(to_512(Image.fromarray(curr_d)))[:, :3]
        
        # audio
        # motion_info = inferer.forward_drving(curr_d)
        # img = inferer.forward_diff(motion_info)

        img = inferer.forward(None, curr_d, crop=False, smooth_pose=False, target_theta=True, mix=False, mix_old=False, modnet_mask=False)
        cv2.imshow('img', np.array(img[0][0])[:,:,::-1])
        img_with_bg = connect_img_and_bg(img[0][0], bg, mdnt=False)
        cv2.imshow('img_with_bg', np.array(img_with_bg)[:,:,::-1])

        # 按 'q' 键退出程序
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    # 释放摄像头
    cap.release()

    # 关闭窗口
    cv2.destroyAllWindows()