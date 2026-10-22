import torch
import numpy as np
from torch.optim import lr_scheduler
import clip
import random
from MNL_Loss import loss_m3
import scipy.stats
from utils import set_dataset_aigc, convert_models_to_fp32, get_logger,log_and_print
from ImageDataset4 import _preprocess2, _preprocess3
import torch.nn.functional as F
from itertools import product
import os
import tqdm
import pandas as pd
from scipy.stats import norm
import loss
import torch.nn as nn

from modules.AMF_Net import GRM_MTD_IQA, set_dataset_aigc_csv


checkpoint_dir = 'checkpoint_rebu3/AGQG_AMF_iqa3k_ep120_1e5'
csv_name = 'AGQG_AMF_iqa3k_ep120_1e5.csv'
session_n = 10

seed = 202006261
torch.manual_seed(seed)     
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 1e-5
num_epoch = 120
bs = 16
train_patch = 8
test_patch = 15
num_workers = 16

is_aigc2023 = False
mtl_dataset = 0
mtl_map = {'AGIQA3K': 0, 'AIGCIQA2023': 1, 'AGIQA1K': 2}



radius = [336, 224, 112]
preprocess2 = [_preprocess2(radius[0]), _preprocess2(radius[1]), _preprocess2(radius[2])]
preprocess3 = [_preprocess3(radius[0]), _preprocess3(radius[1]), _preprocess3(radius[2])]
##############################general setup####################################

def train(model, best_result, best_epoch, mtl):
    with torch.cuda.amp.autocast(enabled=True):
        running_loss = 0
        tt_domain_loss = 0
        diff_total_loss = 0
        model.train()
        loader = train_loaders[0]

        if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
            scheduler.step()
            print(optimizer.state_dict()['param_groups'][0]['lr'])

        avg_loss = 0
        step = -1
        q_mos, q_hat = [],[]
        c_mos, c_hat = [],[]
        a_mos, a_hat = [],[]
        image_names = []  
        loop = tqdm.tqdm(loader, desc='Epoch:{}'.format(epoch))
        
        for sample_batched in loop:
            step += 1
            # ========== loding muti-scale image ==========
            x_l = sample_batched['img_l'].to(torch.float32).to(device)
            x_m = sample_batched['img_m'].to(torch.float32).to(device)
            x_s = sample_batched['img_s'].to(torch.float32).to(device)
            mos_q = sample_batched['mos_q'].to(torch.float32).to(device)
            mos_a = sample_batched['mos_a'].to(torch.float32).to(device)
            mos_c = sample_batched['mos_c'].to(torch.float32).to(device)
            con_tokens = sample_batched['con_tokens'].to(device)
            image_name = sample_batched['image_name'] 


            x_l = x_l.to(torch.float32).to(device)
            x_m = x_m.to(torch.float32).to(device)
            x_s = x_s.to(torch.float32).to(device)
            con_tokens = con_tokens.to(device)

            optimizer.zero_grad()
            # ========== feed forward ==========
            logits_quality = model(x_l, x_m, x_s, con_tokens, require_beta=True)

            logits_per_qua, logits_per_con, logits_per_aes = logits_quality
            loss_q = loss_m3(logits_per_qua, mos_q.detach())     
            loss_c = loss_m3(logits_per_con, mos_c.detach())    

            if mtl == 0:
                total_loss = loss_q + loss_c
            elif mtl == 1:
                loss_a = loss_m3(logits_per_aes, mos_a.detach())
                total_loss = loss_q + loss_a + loss_c
            elif mtl == 2:
                loss_a = loss_m3(logits_per_aes, mos_a.detach())
                total_loss = loss_q + loss_a + loss_c

            loss_fn_2 = loss.plccloss()
            plcc_qua = loss_fn_2(logits_per_qua.unsqueeze(1).float(), mos_q.detach().unsqueeze(1).float())
            plcc_con = loss_fn_2(logits_per_con.unsqueeze(1).float(), mos_c.detach().unsqueeze(1).float())

            if mtl == 1 or mtl == 2:
                plcc_aes = loss_fn_2(logits_per_aes.unsqueeze(1).float(), mos_a.detach().unsqueeze(1).float())
                probs_loss = plcc_qua + plcc_con + plcc_aes
            elif mtl == 0:
                probs_loss = plcc_qua + plcc_con

            total_loss_sum = total_loss + probs_loss
            total_loss_sum.backward()

            image_names.extend(image_name)
            q_mos.extend(mos_c.cpu().numpy())
            q_hat.extend(logits_per_con.detach().cpu().numpy())
            c_mos.extend(mos_q.cpu().numpy())
            c_hat.extend(logits_per_qua.detach().cpu().numpy())
            if mtl == 1 or mtl == 2:
                a_mos.extend(mos_a.cpu().numpy())
                a_hat.extend(logits_per_aes.detach().cpu().numpy())
            elif mtl == 0:
                a_mos.extend([0]*len(mos_q.cpu().numpy()))
                a_hat.extend([0]*len(logits_per_aes.detach().cpu().numpy()))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model.get_clip())

            running_loss += total_loss_sum.data.item()
            avg_loss = running_loss / (step + 1)
            tt_domain_loss += probs_loss.data.item()
            avg_loss2 = tt_domain_loss / (step + 1)

            loop.set_description('Epoch:{}  mse-Loss:{:.4f}  plccmos-Loss:{:.4f}'.format(epoch, avg_loss, avg_loss2))

        log_and_print(base_logger, 'Epoch:{}  mse-Loss:{:.4f}  plccmos-Loss:{:.4f}'.format(epoch, avg_loss, avg_loss2))
        if (epoch >= 0):
            avg_score, avg_plcc, result = eval(aigc_val_loader, phase='val', dataset='live', mtl=mtl)

            if avg_score > best_result['quality']:
                log_and_print(base_logger,'**********New quality best!**********')
                best_epoch['quality'] = epoch
                best_result['quality'] = (avg_score + avg_plcc)/2
                best_result['srcc'] , best_result['plcc'] = avg_score, avg_plcc,
                best_result['srcc_q'], best_result['plcc_q'] = result['srcc_q'] , result['plcc_q']
                best_result['srcc_c'], best_result['plcc_c'] = result['srcc_c'] , result['plcc_c']
                best_result['srcc_a'], best_result['plcc_a'] = result['srcc_a'] ,result['plcc_a']
                dir = os.path.join(checkpoint_dir, str(session + 1))
                os.makedirs(dir,exist_ok = True)
                ckpt_name = os.path.join(checkpoint_dir, str(session + 1), 'quality_best_ckpt.pt')
                torch.save({'model_state_dict': model.state_dict()}, ckpt_name) 
                # df = pd.DataFrame({
                #     'metric': ['quality_srcc', 'quality_plcc', 'consistency_srcc', 'consistency_plcc', 'authenticity_srcc', 'authenticity_plcc'],
                #     'value': [result['srcc_q'], result['plcc_q'], result['srcc_c'], result['plcc_c'], result['srcc_a'], result['plcc_a']],
                #     'epoch': epoch,
                #     'session': session + 1,
                #     'avg': avg_score,
                #     'gt': q_mos,
                #     'pred': q_hat
                # })
                df = pd.DataFrame({
                    'Image Name': image_names,
                    'mos': q_mos, 
                    'mos_pred': q_hat,
                    'authenticity': a_mos, 
                    'authenticity_pred': a_hat,
                    'consistency': c_mos, 
                    'consistency_pred': c_hat,
                })
                df.to_csv(checkpoint_dir+'/train_'+csv_name, index=False)

        return best_result, best_epoch

def eval(loader, phase, dataset, mtl):
    model.eval()

    y_q = []
    y_pred_q = []
    y_a = []
    y_pred_a = []
    y_c = []
    y_pred_c = []
    
    for sample_batched in tqdm.tqdm(loader, desc='{}:{}'.format(dataset, phase)):
        x_l = sample_batched['img_l'].to(torch.float32).to(device)
        x_m = sample_batched['img_m'].to(torch.float32).to(device)
        x_s = sample_batched['img_s'].to(torch.float32).to(device)
        mos_q = sample_batched['mos_q'].to(torch.float32).to(device)
        mos_a = sample_batched['mos_a'].to(torch.float32).to(device)
        mos_c = sample_batched['mos_c'].to(torch.float32).to(device)
        con_tokens = sample_batched['con_tokens'].to(device)

        with torch.no_grad():
            logits_quality = model(x_l, x_m, x_s, con_tokens, require_beta=True)
            logits_per_qua, logits_per_con, logits_per_aes = logits_quality

        y_pred_q.extend(logits_per_qua.cpu().numpy())
        y_pred_a.extend(logits_per_aes.cpu().numpy())
        y_pred_c.extend(logits_per_con.cpu().numpy())
        y_q.extend(mos_q.cpu().numpy())
        y_a.extend(mos_a.cpu().numpy())
        y_c.extend(mos_c.cpu().numpy())

    srcc_q = scipy.stats.mstats.spearmanr(x=y_q, y=y_pred_q)[0]
    plcc_q = scipy.stats.pearsonr(x=y_q, y=y_pred_q)[0]
    srcc_c = scipy.stats.mstats.spearmanr(x=y_c, y=y_pred_c)[0]
    plcc_c = scipy.stats.pearsonr(x=y_c, y=y_pred_c)[0]
    if mtl == 1:
        srcc_a = scipy.stats.mstats.spearmanr(x=y_a, y=y_pred_a)[0]
        plcc_a = scipy.stats.pearsonr(x=y_a, y=y_pred_a)[0]
    else:
        srcc_a, plcc_a = 0, 0

    print_text = dataset + ':' + phase + ': ' +  'qua_srcc:{:.4f}  qua_plcc{:.4f} con_srcc:{:.4f}  con_plcc{:.4f} aes_srcc:{:.4f}  aes_plcc{:.4f}'.format(srcc_q, plcc_q, srcc_c, plcc_c, srcc_a, plcc_a)
    log_and_print(base_logger,print_text)

    result = {
        'srcc_q': srcc_q,
        'plcc_q': plcc_q,
        'srcc_c': srcc_c,
        'plcc_c': plcc_c,
        'srcc_a': srcc_a,
        'plcc_a': plcc_a
    }

    return  (srcc_q + srcc_a + srcc_c) / 3, (plcc_q + plcc_a + plcc_c) / 3, result

if __name__ == '__main__':
    os.makedirs(checkpoint_dir,exist_ok = True)
    best_result_list = []
    base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')

    for session in range(0, session_n):

        model = GRM_MTD_IQA(device=device, clip_net='RN50',in_size=1024, prompt_n=4,D_grm=2.0)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        best_result = { 'quality': 0.0 ,'srcc': 0.0, 'plcc': 0.0}
        best_epoch = {  'quality': 0}

        aigc_train_csv = os.path.join('./Database/AGIQA-3K', str(session+1), 'train.csv')
        aigc_val_csv = os.path.join('./Database/AGIQA-3K', str(session+1), 'val.csv')
        aigc_set = '/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/data/AGIQA-3K'

        # # AIGCIQA2023
        # aigc_train_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'train.csv')
        # aigc_val_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'val.csv')
        # aigc_set = "/third_data/cbl/IQA/zhicheng/AIGCIQA2023/"
        
        aigc_train_loader = set_dataset_aigc_csv(
            csv_file=aigc_train_csv,
            batch_size=bs,
            img_dir=aigc_set,
            num_workers=num_workers,
            preprocess_list=preprocess3,
            patch_num=train_patch,
            test=False,
            mtl=mtl_dataset,
            is_aigc2023=is_aigc2023
        )   
        aigc_val_loader = set_dataset_aigc_csv(
            csv_file=aigc_val_csv,
            batch_size=bs,
            img_dir=aigc_set,
            num_workers=num_workers,
            preprocess_list=preprocess2,
            patch_num=test_patch,
            test=True,
            mtl=mtl_dataset,
            is_aigc2023=is_aigc2023
        )

        train_loaders = [aigc_train_loader]

        result_pkl = {}
        for epoch in range(0, num_epoch):
            best_result, best_epoch = train(model, best_result, best_epoch, mtl=mtl_dataset)
            scheduler.step()
            log_and_print(base_logger,'...............current quality best, session:{}...............'.format(session + 1))
            log_and_print(base_logger,'best quality epoch:{}'.format(best_epoch['quality']))
            log_and_print(base_logger,'best quality result:{}, srcc:{}, plcc{}'.format(best_result['quality'], best_result['srcc'], best_result['plcc']))

        best_result_list.append(best_result)

    avg_srcc = 0
    avg_plcc = 0
    avg_srcc_q = 0
    avg_plcc_q = 0
    avg_srcc_c = 0
    avg_plcc_c = 0
    avg_srcc_a = 0
    avg_plcc_a = 0

    for i in range(session_n):
        avg_srcc += best_result_list[i]['srcc']
        avg_plcc += best_result_list[i]['plcc']
        avg_srcc_q += best_result_list[i]['srcc_q']  
        avg_plcc_q += best_result_list[i]['plcc_q']
        avg_srcc_c += best_result_list[i]['srcc_c']
        avg_plcc_c += best_result_list[i]['plcc_c']
        avg_srcc_a += best_result_list[i]['srcc_a']
        avg_plcc_a += best_result_list[i]['plcc_a']

    avg_srcc = avg_srcc / session_n
    avg_plcc = avg_plcc / session_n
    avg_srcc_q = avg_srcc_q / session_n
    avg_plcc_q = avg_plcc_q / session_n
    avg_srcc_c = avg_srcc_c / session_n
    avg_plcc_c = avg_plcc_c / session_n
    avg_srcc_a = avg_srcc_a / session_n
    avg_plcc_a = avg_plcc_a / session_n
    log_and_print(base_logger,'all_finished,average srcc:{}, plcc:{}'.format(avg_srcc, avg_plcc))
    log_and_print(base_logger,'average qua_srcc:{:.4f}  qua_plcc{:.4f} con_srcc:{:.4f}  con_plcc{:.4f} aes_srcc:{:.4f}  aes_plcc{:.4f}'.format(avg_srcc_q, avg_plcc_q, avg_srcc_c, avg_plcc_c, avg_srcc_a, avg_plcc_a))