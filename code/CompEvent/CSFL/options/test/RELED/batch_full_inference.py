import sys
sys.path.append('/code/CompEvent/CSFL')

import torch
import argparse
import os
import logging
import datetime
from tqdm import tqdm
from basicsr.models import create_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs, tensor2img, imwrite
from basicsr.utils.options import parse
from basicsr.data import create_dataloader, create_dataset
import numpy as np

def init_loggers(opt):
    log_file = os.path.join(opt['path']['log'],
                        f"batch_inference_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    return logger

def create_test_dataloader(opt, logger):
    test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(
                test_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of test images/folders in {dataset_opt["name"]}: '
                f'{len(test_set)}')
        elif phase == 'train':
            logger.info(f'Skipping train phase: {dataset_opt["name"]}')
            continue
        else:
            logger.warning(f'Dataset phase {phase} is not recognized, skipping.')

    if test_loader is None:
        raise ValueError('No test dataset found in configuration.')
    
    return test_loader

def batch_full_inference(opt, logger):

    test_loader = create_test_dataloader(opt, logger)

    model = create_model(opt)

    opt['val']['grids'] = None
    opt['val']['crop_size'] = None

    model.net_g.eval()
    
    logger.info('开始批量全图推理...')

    total_psnr = 0.0
    total_ssim = 0.0
    processed_count = 0

    with torch.no_grad():
        for idx, test_data in enumerate(tqdm(test_loader, desc="处理样本")):
            logger.info(f'处理第 {idx+1}/{len(test_loader)} 个样本...')

            model.feed_data(test_data)

            model.test()

            visuals = model.get_current_visuals()
            sr_imgs = visuals['result']
            if sr_imgs.dim() == 5:
                sr_imgs = sr_imgs[0]
            T = sr_imgs.shape[0]

            if opt['val']['save_img']:
                for t in range(T):
                    sr_img = tensor2img(sr_imgs[t])
                    save_path = os.path.join(
                        opt['path']['visualization'],
                        f'batch_inference_{idx:06d}_frame{t}.png'
                    )
                    imwrite(sr_img, save_path)
                    logger.info(f'结果已保存至: {save_path}')

            if 'metrics' in opt['val'] and 'gt' in visuals:
                from basicsr.metrics import calculate_psnr, calculate_ssim
                gt_imgs = visuals['gt']
                if gt_imgs.dim() == 5:
                    gt_imgs = gt_imgs[0]
                for t in range(T):
                    sr_img = tensor2img(sr_imgs[t])
                    gt_img = tensor2img(gt_imgs[t])
                    psnr = calculate_psnr(sr_img, gt_img, crop_border=opt['val']['crop_border'])
                    ssim = calculate_ssim(sr_img, gt_img, crop_border=opt['val']['crop_border'])
                    total_psnr += psnr
                    total_ssim += ssim
                    processed_count += 1
                    logger.info(f'样本 {idx+1} 帧 {t}: PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')

            del model.lq, model.output
            if hasattr(model, 'gt'):
                del model.gt
            torch.cuda.empty_cache()

    if processed_count > 0:
        avg_psnr = total_psnr / processed_count
        avg_ssim = total_ssim / processed_count
        logger.info(f'批量推理完成！')
        logger.info(f'处理样本数: {processed_count}')
        logger.info(f'平均 PSNR: {avg_psnr:.2f}')
        logger.info(f'平均 SSIM: {avg_ssim:.4f}')
    else:
        logger.warning('没有成功处理任何样本')

def main():

    parser = argparse.ArgumentParser(description='批量全图推理脚本')
    parser.add_argument('-opt', type=str, required=True, help='配置文件路径')

    args = parser.parse_args()

    opt = parse(args.opt, is_train=False)

    opt['val']['grids'] = None
    opt['val']['crop_size'] = None

    make_exp_dirs(opt)

    logger = init_loggers(opt)

    batch_full_inference(opt, logger)

if __name__ == '__main__':
    main()