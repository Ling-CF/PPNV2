import os
import numpy as np
import torch
from PPNV2 import PPNv2
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils import MyDataset, util_of_lpips
os.environ['TORCH_HOME']='/home/Ling.cf/HHD/conda/miniconda3/torch-model'

def show_muti_img(imgs, targets):
    pre_imgs = []
    tar_imgs = []
    for i in range(0,len(imgs),1):
        pre = imgs[i]
        pre = pre.cpu().numpy().squeeze()
        pre = np.transpose(pre, (1, 2, 0))
        pre_imgs.append(pre)
        target = targets[:, i]
        target = target.cpu().squeeze().numpy()
        target = np.transpose(target, (1, 2, 0))
        tar_imgs.append(target)
    stack_img = np.hstack(pre_imgs)
    stack_tar = np.hstack(tar_imgs)
    cv2.imshow('stack_img', stack_img)
    cv2.imshow('tar_img', stack_tar)
    d = cv2.waitKey()
    if d == 27:
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        save_img = input('Save Image? Input "y" if yes: ')
        if save_img == 'y':
            name = input('Input save name: ')
            save_name = './predictions/Caltech/{}_pred.png'.format(name)
            cv2.imwrite(save_name, stack_img*255)
            if input('Save Targets? Input "y" if yes: ') == 'y':
                tar_name = './predictions/Caltech/{}_real.png'.format(name)
                cv2.imwrite(tar_name, stack_tar*255)

def data_process(pred, tar):
    pred = np.transpose(pred, (2, 0, 1))
    tar = np.transpose(tar, (2, 0, 1))
    pred = torch.round(torch.tensor(pred*255))
    pred = torch.tensor(pred, dtype=torch.uint8).unsqueeze(dim=0)
    tar = torch.round(torch.tensor(tar*255))
    tar = torch.tensor(tar, dtype=torch.uint8).unsqueeze(dim=0)
    pred_tar = torch.cat([pred, tar], dim=0)
    pred_tar = pred_tar.unsqueeze(dim=0)
    return pred_tar

def test_model():
    dataset = 'Caltech'
    merge_style = 'modulate'
    test_data='/mnt/DevFiles_NAS/Ling.cf/dataset/{}/test'.format(dataset)                          # the path where validation data is stored
    state_path='./models/{}/pix_{}_DSC_3_big_{}.pt'.format(dataset, dataset, merge_style)        # the path where model state is stored
    num_pre=10                              # the number of time steps in the long term prediction
    show_img=True                           # bool: show image or not (set the batch_size to 1 first if True)
    LenInput = 10                           # length of input sequence

    model = PPNv2(channels=(3, 64, 128, 256, 512, 512), hidden_channels=(3, 64, 128, 256, 512, 512), merge_style=merge_style)
    model = model.to(0)
    checkpoint = torch.load(state_path, map_location='cuda:{}'.format(0))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    all_ssim = []
    all_psnr = []
    all_lpips = []
    lpips_loss = util_of_lpips('alex')
    with torch.no_grad():
        for curdir, dirs, files in os.walk(test_data):
            if len(files) == 0:
                continue
            files.sort()
            # print(curdir)
            for file in files:
                print(file)
                cur_path = os.path.join(curdir, file)
                val_dataset = MyDataset(path=cur_path, len_input=LenInput+num_pre, interval=1, begin=0)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, drop_last=False)
                for data in val_dataloader:
                    inputs = torch.true_divide(data[:, :LenInput], 255).to(0)
                    targets = torch.true_divide(data[:, LenInput:], 255)
                    ssim_score = []
                    psnr_score = []
                    lpips_score = []
                    predictions = model(inputs, pred_steps=num_pre, mode='test')
                    if show_img:
                        show_muti_img(predictions, targets)
                    for t in range(num_pre):
                        target = targets[:, t].to(0)
                        predict = predictions[t]
                        lpips = lpips_loss.calc_lpips(predict, target)
                        lpips_score.append(lpips.mean().item())
                        target = target.squeeze().cpu().numpy()
                        predict = predict.data.cpu().numpy().squeeze()
                        tmp_psnr = []
                        tmp_ssim = []
                        for j in range(target.shape[0]):
                            target_j = target[j, :]
                            predict_j = predict[j, :]
                            if len(target_j.shape) > 2:
                                target_j = np.transpose(target_j, (1, 2, 0))
                                predict_j = np.transpose(predict_j, (1, 2, 0))
                            (ssim, diff) = structural_similarity(target_j, predict_j, win_size=None,
                                                                 multichannel=True, data_range=1.0, full=True)
                            psnr = peak_signal_noise_ratio(target_j, predict_j, data_range=1.0)
                            tmp_psnr.append(psnr)
                            tmp_ssim.append(ssim)
                        psnr_score.append(np.mean(tmp_psnr))
                        ssim_score.append(np.mean(tmp_ssim))
                    # print(ssim_score, '\n', psnr_score, '\n', lpips_score)
                    all_ssim.append(ssim_score)
                    all_psnr.append(psnr_score)
                    all_lpips.append(lpips_score)



    all_ssim = np.array(all_ssim)
    mean_ssim = np.mean(all_ssim, axis=0)
    all_psnr = np.array(all_psnr)
    mean_psnr = np.mean(all_psnr, axis=0)
    all_lpips = np.array(all_lpips)
    mean_lpips = np.mean(all_lpips, axis=0) * 100
    # print('mean ssim: ', '\n', mean_ssim)
    # print("mean psnr: ", '\n', mean_psnr)
    # print("mean lpips: ", '\n', mean_lpips)
    print('mean ssim: ', '\n', mean_ssim, '\n', '0-10: ', np.mean(mean_ssim[:10]), '\n', '0-30: ', np.mean(mean_ssim))
    print("mean psnr: ", '\n', mean_psnr, '\n', '0-10: ', np.mean(mean_psnr[:10]), '\n', '0-30: ', np.mean(mean_psnr))
    print("mean lpips: ", '\n', mean_lpips, '\n', '0-10: ', np.mean(mean_lpips[:10]), '\n', '0-30: ',np.mean(mean_lpips))
    # cla = str(input('classification name: '))
    # ssim_name = './predictions/KTH/metrics/{}_ssim.npy'.format(cla)
    # psnr_name = './predictions/KTH/metrics/{}_psnr.npy'.format(cla)
    # lpips_name = './predictions/KTH/metrics/{}_lpips.npy'.format(cla)
    # np.save(ssim_name, mean_ssim)
    # np.save(psnr_name, mean_psnr)
    # np.save(lpips_name, mean_lpips)





if __name__ == '__main__':
    rank = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    print('start testing')
    test_model()