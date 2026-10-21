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

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

checkpoint_dir = 'inference_original_csv'

dataset = 'agiqa3k'
csv_name = 'TIER_256_2023to3kalign.csv'
score_type = 'align'

os.makedirs(checkpoint_dir,exist_ok = True)

session = 0
weight_path = "/data/cbl/IQA/zhicheng/AGRM/checkpoint_cross/TINR2_resnet50_ori/txt256_2023to3kalign_ep100_nosche/1/quality_best_ckpt.pt"

seed = 202006261
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
bs = 16
train_patch = 8
test_patch = 15
num_workers = 16

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# torch.autograd.set_detect_anomaly(True)

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()


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

        encoded_inputs = tokenizer(prompt,padding='max_length',truncation=True,max_length=256,return_tensors='pt')

        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)

        with torch.no_grad():
            logits_quality, ability, beta= model(x,  input_ids, attention_mask, require_beta=True)    

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
    df.to_csv(checkpoint_dir+'/s'+str(session)+'_'+csv_name, index=False)

    return  (srcc+plcc)/2, srcc, plcc

def eval_original(loader, phase, dataset, score_type):
    model.eval()

    q_mos = []
    q_hat = []
    image_names = []  # 用于存储图片名称
    for sample_batched in tqdm.tqdm(loader, desc='{}:{}'.format(dataset, phase)):

        x, gmos, prompt, image_name = sample_batched['I'], sample_batched[score_type],  sample_batched['prompt'], sample_batched['image_name']
        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist() 

        encoded_inputs = tokenizer(prompt,padding='max_length',truncation=True,max_length=256,return_tensors='pt')

        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)

        I_resized = x[:,-1,:,:,:]
        with torch.no_grad():
            quality_preds, text_emb, img_features = model(I_resized, input_ids, attention_mask) 

        q_hat = q_hat + quality_preds.cpu().tolist()
        image_names.extend(image_name)

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ':' + phase + ': ' +  'srcc:{:.4f}  plcc{:.4f}'.format(srcc,plcc)
    log_and_print(base_logger,print_text)
    df = pd.DataFrame({
        'Image Name': image_names,
        'GMOS': q_mos,
        'prediction': q_hat,
    })
    df.to_csv(checkpoint_dir+'/eval_'+csv_name, index=False)

    return  (srcc+plcc)/2, srcc, plcc

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
best_result_list = []
base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')
if __name__ == '__main__': 

    from modules import TINR2
    # model = TINR2.GRM_TINR(device, D_grm=2)

    model = TINR2.QualityAssessmentModel()

    model.to(device)

    checkpoint = torch.load(weight_path, map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Successfully load ' + weight_path)

    if dataset == 'agiqa1k':
        # aigc_val_csv = os.path.join('./Database/AGIQA-1K', str(session), 'val.csv')
        aigc_val_csv = "./Database/AIGC-1K_All.csv"
        aigc_set = './data/AGIQA-1K/file' #图片路径
        aigc_val_loader = set_dataset_aigc(aigc_val_csv, bs , aigc_set, num_workers, preprocess2, test_patch, True)
    elif dataset == 'agiqa3k':
        # aigc_val_csv = os.path.join('./Database/AGIQA-3K', str(session+1), 'val.csv')
        aigc_val_csv = "./Database/AIGC-3K_All.csv"
        aigc_set = './data/AGIQA-3K' 
        aigc_val_loader = set_dataset_aigc_3k(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)
    elif dataset == 'agiqa2023':
        # aigc_val_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'val.csv')
        aigc_val_csv = "./Database/AIGCIQA2023_All.csv"
        aigc_set = "/data/cbl/IQA/linqing/data/AIGCIQA2023/" 
        aigc_val_loader = set_dataset_aigc_2023(aigc_val_csv, bs , aigc_set, num_workers, preprocess2,test_patch, True)

    # avg_score,srcc,plcc = eval(aigc_val_loader, phase='val', dataset='live', score_type=score_type)

    avg_score,srcc,plcc = eval_original(aigc_val_loader, phase='val', dataset='live', score_type=score_type)
