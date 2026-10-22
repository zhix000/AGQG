# from thop import profile, clever_forma
import clip
import torch.nn as nn
import torch
import pandas as pd
import os

class MLP(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, is_relu=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(ch_in, ch_mid),
            nn.ReLU()
        )
        if is_relu:
            self.fc2 = nn.Sequential(
                nn.Linear(ch_mid, ch_out),
                nn.ReLU()
            )
        else:
            self.fc2 = nn.Sequential(
                nn.Linear(ch_mid, ch_out)
            )

    def forward(self, v):
        v = self.fc1(v)
        v = self.fc2(v)
        return v



class MSAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(in_dim, int(in_dim/2)),
                                 nn.ReLU(),
                                 nn.Linear(int(in_dim/2), in_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度"""
        feat = torch.stack([l, m, s], dim=1)
        attn = self.att(feat)
        attn = self.softmax(attn)
        attn_l, attn_m, attn_s = attn.chunk(3, dim=1)
        attn_l, attn_m, attn_s = attn_l.squeeze(dim=1), attn_m.squeeze(dim=1), attn_s.squeeze(dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s
        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms

from modules.amf_clip_model import load
class MTD_IQA(nn.Module):
    def __init__(self, device, clip_net='RN50', in_size=1024):
        super(MTD_IQA, self).__init__()

        self.in_size = in_size
        # self.base, _ = clip.load(clip_net, device=device)
        self.base = load(clip_net, 'cuda')  
        self.logit_scale = self.base.logit_scale
        self.encode_image = self.base.encode_image
        self.encode_text = self.base.encode_text
        self.MSAMLayer = MSAM(in_dim=in_size)
        self.MLP_q = MLP(ch_in=in_size, ch_mid=256, ch_out=1)
        self.MLP_a = MLP(ch_in=in_size, ch_mid=256, ch_out=1)

    def forward(self, img_l, img_m, img_s, tokens_c):

        tokens_c = tokens_c.view(-1, 77)                        # 2026.03.30

        img_l_feature = self.encode_image(img_l).to(torch.float32)
        img_m_feature = self.encode_image(img_m).to(torch.float32)
        img_s_feature = self.encode_image(img_s).to(torch.float32)

        img_feature = self.MSAMLayer(img_l_feature, img_m_feature, img_s_feature)
        con_feature = self.encode_text(tokens_c).to(torch.float32)

        logits_per_aes = self.MLP_a(img_feature)
        logits_per_qua = self.MLP_q(img_feature)

        # logit_scale = self.logit_scale.exp()
        logit_scale = self.logit_scale.exp() / 10
        img_feature_n = img_feature / img_feature.norm(dim=1, keepdim=True)
        con_feature_n = con_feature / con_feature.norm(dim=1, keepdim=True)
        con_feature_n = con_feature_n.view(-1, 1, self.in_size)
        img_feature_n = img_feature_n.view(-1, 1, self.in_size)
        logits_per_con = [logit_scale * img_feature_n[i] @ con_feature_n[i].t() for i in range(con_feature_n.shape[0])]
        logits_per_con = torch.cat(logits_per_con, dim=0)

        # logits_per_con = 100* ( img_feature_n @ con_feature_n.t() )
        # logits_per_con = torch.diagonal(logits_per_con).unsqueeze(1)

        return logits_per_qua, logits_per_con, logits_per_aes, img_feature, con_feature

    def get_clip(self):
        return self.base

def grm_sigm(theta, beta, alpha=1, D=1.702):       # 2025.03.03 
    # 计算每个等级的概率
    probabilities = []
    for k in range(beta.size(1) + 1):
        if k == 0:
            # 使用 torch.sigmoid 替代手动计算
            p_k = 1 - torch.sigmoid(D * alpha * (theta - beta[:, 0]))
        elif k == beta.size(1):
            p_k = torch.sigmoid(D * alpha * (theta - beta[:, -1]))
        else:
            # 使用 torch.sigmoid 替代手动计算
            p_k = torch.sigmoid(D * alpha * (theta - beta[:, k-1])) - torch.sigmoid(D * alpha * (theta - beta[:, k]))
        probabilities.append(p_k)
    # 将概率列表转换为张量
    probabilities_tensor = torch.stack(probabilities, dim=1)
    return probabilities_tensor

class MLP2(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=2, out_dim=2,output_activation='tanh'):
        super(MLP2, self).__init__()
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()
        self.layers = nn.ModuleList()

        ch_in = input_dim
        for i in range(num_layers):
            self.layers.append(nn.Linear(ch_in, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.layers.append(
                self.activation_1 if i < (num_layers - 1) else self.activation_2
            )
            ch_in = hidden_dim  # 后续层的输入维度等于 hidden_dim

        self.layers.append(nn.Linear(ch_in, out_dim))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class TemperaturedTeLU_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, temperature):
        return (x+temperature)*torch.tanh(torch.exp(temperature + x))


# AGRM评分模块，对输入的文本和图像向量映射得到参数值，然后使用GRM进行得到质量评分
class AGQG(nn.Module):
    def  __init__(self, device, img_in_dim=1024, txt_in_dim=1024 ,prompt_n=4, D_grm=2.0):
        super(AGQG, self).__init__()
        self.img_proj = MLP2(input_dim=img_in_dim, out_dim=1)   
        self.txt_proj = MLP2(input_dim=txt_in_dim, out_dim=2)
        self.hardsigmoid = TemperaturedTeLU_2()
        self.prompt_n = prompt_n
        self.device = device
        self.D_grm = D_grm

    def forward(self, img_feature, txt_feature, ability, require_beta=False):
        params = self.img_proj(img_feature)     
        txt_params = self.txt_proj(txt_feature)    

        tp1, tp2 = params[:, 0], params[:, 0]
        start = self.hardsigmoid(txt_params[:, 0], temperature=tp1) 
        delta = self.hardsigmoid(txt_params[:, 1], temperature=tp2) + 1.2
        beta = start.unsqueeze(1) + (torch.arange(self.prompt_n, device=self.device)).float() * delta.unsqueeze(1)
        logits_k = grm_sigm(ability, beta=beta, D=self.D_grm)

        level_n = self.prompt_n + 1
        quality_preds = torch.matmul(logits_k, torch.arange(1, level_n + 1, dtype=logits_k.dtype).to(self.device))
        quality_preds = ((quality_preds -1) / (level_n - 1)) * 5
        if require_beta:
             return quality_preds, logits_k, ability, beta
        return quality_preds


# 核心模型：GRM_MTD_IQA
class GRM_MTD_IQA(nn.Module):
    def __init__(self, device, clip_net='RN50', in_size=1024, prompt_n=4, depth=6, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      
        self.D_grm = D_grm
        self.in_size = in_size
        
        # 1. MTD_IQA多任务特征提取
        self.mtd_iqa = MTD_IQA(device=device, clip_net=clip_net, in_size=in_size)
        
        # 2. 三维度AGQG分支
        self.AGQG1 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)
        self.AGQG2 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)
        self.AGQG3 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)

    def forward(self, x_l, x_m, x_s, input_token_c, require_beta=False):
        # MTD_IQA前向：输出三维度logits + 图像/文本特征
        # import pdb;pdb.set_trace()
        input_token_c = input_token_c.view(-1, 77)
        logits_qua, logits_con, logits_aes, img_feature, con_feature  = self.mtd_iqa(x_l, x_m, x_s, input_token_c)
        
        # 计算三维度能力值
        ability_qua = logits_qua.squeeze(1)  # [batch_size] 质量能力值
        ability_aes = logits_aes.squeeze(1)  # [batch_size] 美学能力值
        ability_con = logits_con.squeeze(1) / 10# [batch_size] 一致性能力值

        # AGQG预测GRM分数
        quality_qua = self.AGQG1(img_feature, con_feature, ability_qua)  # [B, prompt_n]
        quality_con = self.AGQG2(img_feature, con_feature, ability_con)  # [B, prompt_n]
        quality_aes = self.AGQG3(img_feature, con_feature, ability_aes)  # [B, prompt_n]
        
        return quality_qua, quality_con, quality_aes

    def get_clip(self):
        """适配原代码的convert_models_to_fp32逻辑"""
        return self.mtd_iqa.base

class GRM_MTD_IQA_one(nn.Module):
    def __init__(self, device, clip_net='RN50', in_size=1024, prompt_n=4, D_grm=1.7, model_type='qua'):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      
        self.D_grm = D_grm
        self.in_size = in_size
        self.model_type = model_type
        # 1. MTD_IQA多任务特征提取
        self.mtd_iqa = MTD_IQA(device=device, clip_net=clip_net, in_size=in_size)
        
        # 2. 一维度AGQG分支
        self.AGQG1 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)

    def forward(self, x_l, x_m, x_s, input_token_c, require_beta=False):
        # MTD_IQA前向：输出三维度logits + 图像/文本特征
        # import pdb;pdb.set_trace()
        # input_token_c = input_token_c.view(-1, 77)
        logits_qua, logits_con, logits_aes, img_feature, con_feature  = self.mtd_iqa(x_l, x_m, x_s, input_token_c)
        
        # 计算三维度能力值
        ability_qua = logits_qua.squeeze(1)  # [batch_size] 质量能力值
        ability_aes = logits_aes.squeeze(1)  # [batch_size] 美学能力值
        # ability_con = logits_con.squeeze(1) / 10# [batch_size] 一致性能力值
        ability_con = logits_con.squeeze(1)

        if self.model_type == 'qua':
            ability = ability_qua
        elif self.model_type == 'con':
            ability = ability_con
        elif self.model_type == 'aes':
            ability = ability_aes    
        else:
            ability = ability_qua + ability_aes + ability_con
        # AGQG预测GRM分数
        quality_preds, logits_k, ability, beta = self.AGQG1(img_feature, con_feature, ability, require_beta=True)  # [B, prompt_n]
        
        if require_beta:
             return quality_preds, logits_k, ability, beta
        return quality_preds

    def get_clip(self):
        """适配原代码的convert_models_to_fp32逻辑"""
        return self.mtd_iqa.base

class GRM_MTD_IQA_Agg(nn.Module):
    def __init__(self, device, clip_net='RN50', in_size=1024, prompt_n=4, depth=6, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      
        self.D_grm = D_grm
        self.in_size = in_size
        
        # 1. MTD_IQA多任务特征提取
        self.mtd_iqa = MTD_IQA(device=device, clip_net=clip_net, in_size=in_size)
        
        # 2. 三维度AGQG分支
        self.AGQG1 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)
        self.AGQG2 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)
        self.AGQG3 = AGQG(device, img_in_dim=in_size, txt_in_dim=in_size, prompt_n=prompt_n, D_grm=D_grm)

    def forward(self, x_l, x_m, x_s, input_token_c, require_beta=False):
        # MTD_IQA前向：输出三维度logits + 图像/文本特征
        # import pdb;pdb.set_trace()
        # input_token_c = input_token_c.view(-1, 77)
        logits_qua, logits_con, logits_aes, img_feature, con_feature  = self.mtd_iqa(x_l, x_m, x_s, input_token_c)
        
        # 计算三维度能力值
        ability_qua = logits_qua.squeeze(1)  # [batch_size] 质量能力值
        ability_aes = logits_aes.squeeze(1)  # [batch_size] 美学能力值
        ability_con = logits_con.squeeze(1) / 10# [batch_size] 一致性能力值

        # AGQG预测GRM分数
        quality_qua, logits_k, ability, beta = self.AGQG1(img_feature, con_feature, ability_qua, require_beta=True)  # [B, prompt_n]
        quality_con = self.AGQG2(img_feature, con_feature, ability_con)  # [B, prompt_n]
        quality_aes = self.AGQG3(img_feature, con_feature, ability_aes)  # [B, prompt_n]
        
        quality_preds = quality_qua + quality_con + quality_aes

        if require_beta:
             return quality_preds, logits_k, ability, beta
        return quality_preds

    def get_clip(self):
        """适配原代码的convert_models_to_fp32逻辑"""
        return self.mtd_iqa.base


# ====================== 2. 新增CSV版数据集类（多尺度） ======================
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def image_loader(image_name):
    from PIL import Image
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')

def get_default_img_loader():
    import functools
    return functools.partial(image_loader)

def get_class_names(name, mtl):
    dir_txt = {0: './IQA_Database/AGIQA-3K_index.txt',
               1: './IQA_Database/AIGCIQA2023_index.txt',
               3: './IQA_Database/PKU-I2IQA_index.txt'}

    textpath = dir_txt[mtl]
    with open(textpath, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    f.close()
    all_name = []
    all_class = []
    for i in lines:
        i_name, i_class = i.split('\t')[1], i.split('\t')[0]
        all_name.append(i_name)
        all_class.append(i_class)
    select_idx = all_name.index(name)
    class_id = all_class[select_idx]
    return class_id

class AIGCIQA_CSV_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, 
                 img_dir, 
                 mtl, 
                 preprocess,  # 多尺度预处理列表 [preprocess_l, preprocess_m, preprocess_s]
                 test, 
                 is_aigc2023=False,
                 get_loader=get_default_img_loader, 
                 get_class=False):
        # 1. 读取CSV文件
        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]  # 跳过表头
        self.data.reset_index(drop=True, inplace=True)
        print(f'%d csv data successfully loaded!' % len(self.data))

        # 2. 解析CSV到列表
        self.img_paths = []
        self.all_names = []
        self.mos1 = []    # Quality
        self.mos2 = []    # Authenticity
        self.mos3 = []    # Correspondence
        self.con_text_prompts = []
        self.score_type_map = {'mos':'quality', 'authenticity':'authenticity', 'corr':'correspondence', 'align':'correspondence'}

        self.mtl = mtl
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            if self.mtl==1:
                # AIGCIQA2023列：[模型名, 图像名, 质量, 真实性, 一致性, Prompt]
                model_dir = row[0]
                img_name = row[1]
                img_path = os.path.join(img_dir, model_dir, img_name)
                mos_q = float(row[2])
                mos_a = float(row[3])
                mos_c = float(row[4])
                prompt = row[5]
            elif self.mtl==0:
                # AGIQA3K列：[图像路径, 质量, 一致性, Prompt]
                img_path = os.path.join(img_dir, row[0])
                img_name = os.path.basename(row[0])
                mos_q = float(row[5])
                mos_a = 0.0
                mos_c = float(row[7])
                prompt = row[1]
            elif self.mtl==2:
                # AGIQA1K列：[图像路径, Prompt, 质量]
                img_path = os.path.join(img_dir, row[0])
                img_name = img_path
                mos_q = float(row[2])
                mos_a = 0.0
                mos_c = 0.0
                prompt = row[1]

            self.img_paths.append(img_path)
            self.all_names.append(img_name)
            self.mos1.append(mos_q)
            self.mos2.append(mos_a)
            self.mos3.append(mos_c)
            self.con_text_prompts.append([prompt])

        # 3. 初始化参数
        self.img_dir = img_dir
        self.get_class = get_class
        self.loader = get_loader()
        self.preprocess = preprocess  # [l, m, s]多尺度预处理
        self.test = test
        # self.is_aigc2023 = is_aigc2023

    def __getitem__(self, index):
        # 1. 读取图像
        image_path = self.img_paths[index]
        I = self.loader(image_path)

        # 2. 多尺度预处理
        I_l = self.preprocess[0](I)  # 336×336
        I_m = self.preprocess[1](I)  # 224×224
        I_s = self.preprocess[2](I)  # 112×112

        # 3. Prompt token化
        con_text_prompts = self.con_text_prompts[index]
        con_tokens = torch.cat([clip.tokenize(prompt) for prompt in con_text_prompts])

        # 4. 读取分数
        mos_q = self.mos1[index]
        mos_a = self.mos2[index]
        mos_c = self.mos3[index]

        # 5. 构造样本（兼容原字段）
        sample = {
            'img_l': I_l, 'img_m': I_m, 'img_s': I_s,
            'mos_q': mos_q, 'mos_a': mos_a, 'mos_c': mos_c,
            'con_tokens': con_tokens, 'image_name': self.all_names[index],
            'prompt': con_text_prompts[0],  # 兼容原prompt字段
            'mos': mos_q, 'authenticity': mos_a, 'corr': mos_c, 'align': mos_c  # 兼容原score_type字段
        }

        if self.get_class:
            sample['class'] = get_class_names(self.all_names[index], self.mtl)

        return sample

    def __len__(self):
        return len(self.data)

# ====================== 3. 数据集加载函数（适配多尺度CSV数据集） ======================
def set_dataset_aigc_csv(csv_file, batch_size, img_dir, num_workers, preprocess_list, patch_num, test, mtl=1, is_aigc2023=True):
    """
    替换原set_dataset_aigc_2023/3k，返回多尺度CSV数据集的DataLoader
    """
    dataset = AIGCIQA_CSV_Dataset(
        csv_file=csv_file,
        img_dir=img_dir,
        mtl=mtl,
        preprocess=preprocess_list,
        test=test,
        is_aigc2023=is_aigc2023
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not test,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     batch_size = 1
#     text_c = ["a beautiful women"]*batch_size
#     token_c = torch.stack([clip.tokenize(prompt) for prompt in text_c]).to(device)
#     x_m = torch.randn(batch_size, 3, 224, 224).to(device)
#     x_l = torch.randn(batch_size, 3, 336, 336).to(device)
#     x_s = torch.randn(batch_size, 3, 112, 112).to(device)

#     model = MTD_IQA(device=device).to(device)
#     pred_c = []
#     pred_q = []
#     pred_a = []
#     pred_nss = []

#     input_token_c = token_c.view(-1, 77)

#     for param in model.parameters():
#         param.requires_grad = True
#     flops, params = profile(model, inputs=(x_l, x_m, x_s, input_token_c))
#     print('Flops: % .4fG' % (2 * flops / 1000000000))
#     print('params: % .4fM' % (params / 1000000))
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    prompt_n = 4
    
    # 构造输入
    x_l = torch.randn(batch_size, 3, 336, 336).to(device)
    x_m = torch.randn(batch_size, 3, 224, 224).to(device)
    x_s = torch.randn(batch_size, 3, 112, 112).to(device)
    text_c = ["a beautiful women"]*batch_size
    input_token_c = torch.stack([clip.tokenize(prompt) for prompt in text_c]).view(-1, 77).to(device)
    
    # 初始化模型
    model = GRM_MTD_IQA(device=device, prompt_n=prompt_n, D_grm=2.0).to(device)
    
    # 前向传播
    quality_qua, quality_con, quality_aes = model(x_l, x_m, x_s, input_token_c, require_beta=True)
    
    print(quality_qua)
    print(quality_con)
    print(quality_aes)