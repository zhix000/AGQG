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

checkpoint_dir = 'inference_all_csv'

dataset = 'agiqa3k'
csv_name = 'clipiqa_2023corr.csv'
score_type = 'align'

os.makedirs(checkpoint_dir,exist_ok = True)


session = 0
seed = 202006261
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
 
        with torch.no_grad():
            logits_quality, ability, beta= model(x,  prompt, require_beta=True)


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
    df.to_csv(checkpoint_dir+'/s'+str(session)+csv_name, index=False)
    # 3. 计算整个文件的方差
    row_means = df[['beta1', 'beta2', 'beta3', 'beta4']].mean(axis=1)
    variance = row_means.var(ddof=1)  # 样本方差
    log_and_print(base_logger, "\n 整个文件的方差（样本方差）：{}".format(variance))

    return  (srcc+plcc)/2, srcc, plcc


best_result_list = []
base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')
if __name__ == '__main__': 

    # 存储所有 session 的 SRCC / PLCC
    srcc_all = []
    plcc_all = []

    # GRM-IPCE
    from modules import modelzoos, clip_agiqa2, clipiqa_arch, clipiqa_align
    # model = modelzoos.GRM_0810(device, D_grm=2)
    # 噪声文本的消融实验
    from modules import ipce_ablation
    # model = ipce_ablation.AGQG_IPCE_noise_prompt(device, D_grm=2, corruption_type='missing', action_type='substitute', noise_level=0.3)
    model = ipce_ablation.AGQG_IPCE_noise_prompt(device, D_grm=2, corruption_type='typo', action_type='insert', noise_level=0.05)
    # model = ipce_ablation.AGQG_IPCE_noise_prompt(device, D_grm=2, corruption_type='typo', action_type='delete', noise_level=0.3)
    # model = ipce_ablation.AGQG_IPCE_noise_prompt(device, D_grm=2, corruption_type='typo', action_type='substitute', noise_level=0.3)
    # model = ipce_ablation.AGQG_IPCE_noise_prompt(device, D_grm=2, corruption_type='word_removal', action_type='substitute', noise_level=0.1)
    # model = ipce_ablation.AGQG_IPCE_noise_prompt(device, D_grm=2, corruption_type='word_removal', action_type='substitute', noise_level=0.3)
    # 原模型噪声文本的消融实验
    # model = ipce_ablation.IPCE_noise_prompt(device, D_grm=2, corruption_type='typo', action_type='insert', noise_level=0.3)


    # -----------------------GRM-CLIIQA----------------------
    # 评价mos和aut
    # model = clipiqa_arch.GRM_clipiqa2(device)
    # 评价align
    # model = clipiqa_align.GRM_clipiqa2(device)
    # --------------------------------------------------------

    # -----------------------GRM-CLIP-AGIQA----------------------
    # model = clip_agiqa2.GRM_aigciqa2(device)
    # --------------------------------------------------------

    model.to(device)

    # 存储所有 session 的 SRCC / PLCC
    srcc_all = []
    plcc_all = []

    for session in range(0, 10):  

        # weight_path = "/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/checkpoint_sota/0810_prompt_clip_Telu2_D2_plcc_2023aut_ep100/1/quality_best_ckpt.pt"
        weight_path = f"/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/checkpoint_sota/0810_clip_Telu2_D2_plcc_3kalign_ep100/{session+1}/quality_best_ckpt.pt"

        checkpoint = torch.load(weight_path, map_location=device) 

        model.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully load ' + weight_path)

        if dataset == 'agiqa1k':
            aigc_val_csv = os.path.join('./Database/AGIQA-1K', str(session), 'val.csv')
            # aigc_val_csv = "/home/zhicheng/IQA/AGRM/AGRM1/Database/AIGC-1K_All.csv"
            aigc_set = './data/AGIQA-1K/file' #图片路径
            aigc_val_loader = set_dataset_aigc(aigc_val_csv, bs , aigc_set, num_workers, preprocess2, test_patch, True)
        elif dataset == 'agiqa3k':
            aigc_val_csv = os.path.join('./Database/AGIQA-3K', str(session+1), 'val.csv')
            # aigc_val_csv = "/home/zhicheng/IQA/AGRM/AGRM1/Database/AIGC-3K_All.csv"
            aigc_set = '/third_data/cbl/IQA/zhicheng/AGRM/IPCE-std/data/AGIQA-3K'
            aigc_val_loader = set_dataset_aigc_3k(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)
        elif dataset == 'agiqa2023':
            aigc_val_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'val.csv')
            # aigc_val_csv = "/home/zhicheng/IQA/AGRM/AGRM1/Database/AIGCIQA2023_All.csv"
            aigc_set = "/third_data/cbl/IQA/zhicheng/AIGCIQA2023/"
            aigc_val_loader = set_dataset_aigc_2023(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)

        # GRM—IPCE和CLIP—AGIQA使用混合精度
        with torch.cuda.amp.autocast(enabled=True):
            avg_score,srcc,plcc = eval(aigc_val_loader, phase='val', dataset='live', score_type=score_type)

        # 保存当前 session 结果
        srcc_all.append(srcc)
        plcc_all.append(plcc)

        # 打印当前 session
        print(f"\n📊 Session {session+1} 结果：")
        print(f"SRCC: {srcc:.4f} | PLCC: {plcc:.4f}")

    # ====================== 🔥 所有 session 跑完，计算平均值 ======================
    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)

    srcc_std = np.std(srcc_all, ddof=1)
    plcc_std = np.std(plcc_all, ddof=1)

    print("\n\n" + "="*60)
    print("           🔥 全部 10 个 Session 最终结果 🔥")
    print("="*60)
    print(f"📈 SRCC 均值: {srcc_mean:.4f} ± {srcc_std:.4f}")
    print(f"📈 PLCC 均值: {plcc_mean:.4f} ± {plcc_std:.4f}")
    print("="*60)