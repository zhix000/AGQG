import torch
import numpy as np
from torch.optim import lr_scheduler
import clip_atten
from clip_atten import clip
import random
from MNL_Loss import loss_m3
import scipy.stats
from utils import set_dataset_aigc_2023, _preprocess2, _preprocess3, convert_models_to_fp32, get_logger,log_and_print,set_dataset_aigc_3k
from utils import set_dataset_aigc
import torch.nn.functional as F
from itertools import product
import os
import tqdm
import pandas as pd
from scipy.stats import norm
import loss


checkpoint_dir = 'checkpoint/agqa_aigciqa2_2023to3kmos_ep100'
csv_name = 'agqa_aigciqa2_2023to3kmos_ep100.csv'
score_type = 'mos'
session_n = 10

os.makedirs(checkpoint_dir,exist_ok = True)
seed = 202006261
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 1e-5
num_epoch = 100
bs = 16
train_patch = 8
test_patch = 15
num_workers = 16

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

def deco_prompt(prompt, score_type):
    for pindex in range(len(prompt)):
        if score_type == 'mos':
            prompt[pindex] = 'A photo of good quality and clear details'  
        elif score_type == 'corr' or score_type == 'align':  
            prompt[pindex] = prompt[pindex]  
        elif score_type == 'authenticity':
            prompt[pindex] = 'A photo with genuine scene content and no synthetic artifacts.'
        else:
            print('score_type error')                                  
    return prompt

def train(model, best_result, best_epoch, score_type):
    with torch.cuda.amp.autocast(enabled=True):

        running_loss = 0
        tt_domain_loss = 0
        model.train()
        loader = train_loaders[0]

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
            scheduler.step()
            print(optimizer.state_dict()['param_groups'][0]['lr'])

        avg_loss = 0
        step = -1
        q_mos = []
        q_hat = []
        batch_logits = []  # for storing the prediction logits of each batch
        image_names = []   # for storing the image names
        batch_ability = []
        batch_beta = []
        loop = tqdm.tqdm(loader, desc='Epoch:{}'.format(epoch))
        for sample_batched in loop:
            step += 1
            x, prompt, image_name = sample_batched['I'],  sample_batched['prompt'], sample_batched['image_name']
            x = x.to(device)

            # Intra-dataset setting
            gmos = sample_batched[score_type]

            # Cross-dataset setting
            # corr present in aigc2023, but not present in 3k, so we need to use align as the target for training
            # if score_type == 'corr':
            #     gmos = sample_batched['align']
            # elif score_type == 'align':
            #     gmos = sample_batched['corr']
            # else: 
            #     gmos = sample_batched[score_type]

            prompt = deco_prompt(prompt, score_type)
            gmos = gmos.to(device)
            ori_texts = torch.cat([clip.tokenize(c, truncate=True) for c in prompt]).to(device)

            optimizer.zero_grad()
            logits_quality, ability, beta= model(x,  ori_texts, require_beta=True)

            batch_logits.extend(logits_quality.cpu().tolist())
            image_names.extend(image_name)
            batch_ability.extend(ability.cpu().tolist())
            batch_beta.extend(beta.cpu().tolist())

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            logits_quality = ((logits_quality -1) / 4) *5

            mse_loss = loss_m3(logits_quality, gmos.detach()).mean()        


            loss_fn_2 = loss.plccloss()
            probs_loss = loss_fn_2(logits_quality.unsqueeze(1).float(), gmos.detach().unsqueeze(1).float())

            total_loss_sum = mse_loss + probs_loss
            total_loss_sum.backward()

            q_mos = q_mos + gmos.cpu().tolist() 
            q_hat = q_hat + logits_quality.cpu().tolist()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model.get_clip())
                optimizer.step()
                clip_atten.model.convert_weights(model.get_clip())

            running_loss += mse_loss.data.item()
            avg_loss = running_loss / (step + 1)
            tt_domain_loss += probs_loss.data.item()
            avg_loss2 = tt_domain_loss / (step + 1)

            loop.set_description('Epoch:{}  mse-Loss:{:.4f}  reg-Loss:{:.4f}'.format(epoch, avg_loss, avg_loss2))

        if (epoch >= 0):
            avg_score,srcc,plcc = eval(aigc_val_loader, phase='val', dataset='live', score_type=score_type)

            if avg_score > best_result['quality']:
                log_and_print(base_logger,'**********New quality best!**********')
                best_epoch['quality'] = epoch
                best_result['quality'] = avg_score
                best_result['srcc'] = srcc
                best_result['plcc'] = plcc
                dir = os.path.join(checkpoint_dir, str(session + 1))
                os.makedirs(dir,exist_ok = True)
                ckpt_name = os.path.join(checkpoint_dir, str(session + 1), 'quality_best_ckpt.pt')
                torch.save({'model_state_dict': model.state_dict()}, ckpt_name) 
                df = pd.DataFrame({
                    'Image Name': image_names,
                    'GMOS': q_mos, 'prediction': q_hat,
                    'Logit1': [logit[0] for logit in batch_logits],'Logit2': [logit[1] for logit in batch_logits],
                    'Logit3': [logit[2] for logit in batch_logits],'Logit4': [logit[3] for logit in batch_logits],
                    'Logit5': [logit[4] for logit in batch_logits],
                    'ability': batch_ability,
                    'beta1': [logit[0] for logit in batch_beta],'beta2': [logit[1] for logit in batch_beta],
                    'beta3': [logit[2] for logit in batch_beta],'beta4': [logit[3] for logit in batch_beta],
                })
                df.to_csv(checkpoint_dir+'/train_'+csv_name, index=False)
 
        return best_result, best_epoch


def eval(loader, phase, dataset, score_type):
    model.eval()

    q_mos = []
    q_hat = []
    batch_logits = []  # 用于存储每个批次的预测结果
    image_names = []  # 用于存储图片名称
    batch_ability = []
    batch_beta = []
    for sample_batched in tqdm.tqdm(loader, desc='{}:{}'.format(dataset, phase)):

        x, gmos, prompt, image_name = sample_batched['I'], sample_batched[score_type],  sample_batched['prompt'], sample_batched['image_name']
        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist() 

        prompt = deco_prompt(prompt, score_type)  # 2024.02.04

        ori_texts = torch.cat([clip.tokenize(c, truncate=True) for c in prompt]).to(device) 

        with torch.no_grad():
            logits_quality, ability, beta= model(x,  ori_texts, require_beta=True)

        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                    4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        quality_preds = ((quality_preds -1) / 4) *5

        q_hat = q_hat + quality_preds.cpu().tolist()
        # import pdb;pdb.set_trace()
        batch_logits.extend(logits_quality.cpu().tolist())
        image_names.extend(image_name)
        batch_ability.extend(ability.cpu().tolist())
        batch_beta.extend(beta.cpu().tolist())

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ':' + phase + ': ' +  'srcc:{:.4f}  plcc{:.4f}'.format(srcc,plcc)
    log_and_print(base_logger,print_text)
    # if session == 0:
    df = pd.DataFrame({
        'Image Name': image_names,
        'GMOS': q_mos,
        'prediction': q_hat,
        'Logit1': [logit[0] for logit in batch_logits],
        'Logit2': [logit[1] for logit in batch_logits],
        'Logit3': [logit[2] for logit in batch_logits],
        'Logit4': [logit[3] for logit in batch_logits],
        'Logit5': [logit[4] for logit in batch_logits],
        'ability': batch_ability,
        'beta1': [logit[0] for logit in batch_beta],
        'beta2': [logit[1] for logit in batch_beta],
        'beta3': [logit[2] for logit in batch_beta],
        'beta4': [logit[3] for logit in batch_beta],
    })
    df.to_csv(checkpoint_dir+'/eval_'+csv_name, index=False)

    return  (srcc+plcc)/2, srcc, plcc


best_result_list = []
base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')

for session in range(0, session_n): 
    from modules import clip_agiqa2

    model = clip_agiqa2.GRM_aigciqa2(device, D_grm=2)
    # 使用全Prompt的CLIP-AGIQA
    # model = clip_agiqa2.GRM_aigciqa2(device, D_grm=2, use_clipagiqa_full_prompt=6)
    # Learneable Thresholds
    # model = clip_agiqa2.GRM_aigciqa2_Learnable(device, D_grm=2)
    # Geometric Sequence
    # model = clip_agiqa2.GRM_aigciqa2_geometric(device, D_grm=2, K=1.5)
    # model = clip_agiqa2.GRM_aigciqa2_geometric(device, D_grm=2, K=2.0)
    # Cumulative Softmax
    # model = clip_agiqa2.GRM_aigciqa2_softmax(device, D_grm=2)
    # Gaussian CDF quantiles
    # model = clip_agiqa2.GRM_aigciqa2_CDF(device, D_grm=2)
    # hyperparameter sensitivity
    # model = clip_agiqa2.GRM_aigciqa2(device, D_grm=3.0)
    # model = clip_agiqa2.GRM_aigciqa2(device, D_grm=2, eta=2.0)


    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.001)

    model.to(device)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    train_loss = []
    start_epoch = 0
    best_result = { 'quality': 0.0 ,'srcc': 0.0, 'plcc': 0.0}
    best_epoch = {  'quality': 0}

    # aigc_train_csv = os.path.join('./Database/AGIQA-1K', str(session), 'train.csv')
    # aigc_val_csv = os.path.join('./Database/AGIQA-1K', str(session), 'val.csv')
    # aigc_set = './data/AGIQA-1K/file' #图片路径
    # aigc_train_loader = set_dataset_aigc(aigc_train_csv, bs, aigc_set, num_workers, preprocess3, train_patch, False)   
    # aigc_val_loader = set_dataset_aigc(aigc_val_csv, bs , aigc_set, num_workers, preprocess2, test_patch, True)

    # aigc_train_csv = os.path.join('./Database/AGIQA-3K', str(session+1), 'train.csv')
    # aigc_val_csv = os.path.join('./Database/AGIQA-3K', str(session+1), 'val.csv')
    # aigc_set = '/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/data/AGIQA-3K'
    # aigc_train_loader = set_dataset_aigc_3k(aigc_train_csv, bs, aigc_set, num_workers, preprocess3, train_patch, False)   
    # aigc_val_loader = set_dataset_aigc_3k(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)

    aigc_train_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'train.csv')
    aigc_val_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'val.csv')
    aigc_set = "/third_data/cbl/IQA/zhicheng/AIGCIQA2023/" 
    aigc_train_loader = set_dataset_aigc_2023(aigc_train_csv, bs, aigc_set, num_workers, preprocess3, train_patch, False)   
    aigc_val_loader = set_dataset_aigc_2023(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)

    # aigc_train_loader = set_dataset_aigc_3k('./Database/AIGC-3K_All.csv', bs, '/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/data/AGIQA-3K', 
    #                                             num_workers, preprocess3, train_patch, False) 
    # aigc_set = "/third_data/cbl/IQA/zhicheng/AIGCIQA2023/" 
    # aigc_val_loader = set_dataset_aigc_2023('./Database/AIGCIQA2023_All.csv', bs , aigc_set, num_workers, preprocess2, test_patch, True)

    # aigc_set = "/third_data/cbl/IQA/zhicheng/AIGCIQA2023/" 
    # aigc_train_loader = set_dataset_aigc_2023('./Database/AIGCIQA2023_All.csv', bs, aigc_set, num_workers, preprocess3, train_patch, False) 
    # aigc_val_loader = set_dataset_aigc_3k('./Database/AIGC-3K_All.csv', bs, '/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/data/AGIQA-3K', 
    #                                             num_workers, preprocess3, train_patch, True) 

    train_loaders = [aigc_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch = train(model, best_result, best_epoch, score_type = score_type)
        scheduler.step()
        log_and_print(base_logger,'...............current quality best, session:{}...............'.format(session + 1))
        log_and_print(base_logger,'best quality epoch:{}'.format(best_epoch['quality']))
        log_and_print(base_logger,'best quality result:{}, srcc:{}, plcc{}'.format(best_result['quality'], best_result['srcc'], best_result['plcc']))

    best_result_list.append(best_result)

avg_srcc = 0
avg_plcc = 0
for i in range(session_n):
    avg_srcc += best_result_list[i]['srcc']
    avg_plcc += best_result_list[i]['plcc']

avg_srcc = avg_srcc / session_n
avg_plcc = avg_plcc / session_n
log_and_print(base_logger,'all_finished,average srcc:{}, plcc:{}'.format(avg_srcc, avg_plcc))