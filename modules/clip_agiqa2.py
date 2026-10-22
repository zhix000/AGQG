import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip_atten import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

import os
def load_clip_to_cpu():
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS["ViT-B/16"]
    model_path = clip._download(url,os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_ori=False):
 
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.transformer(x)                      #clip_atten中的clip返回了注意力矩阵
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        # return x
        x = x @ self.text_projection
        x_max = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        if return_ori == False:
            return x_max
        return x_max, x, tokenized_prompts.argmax(dim=-1)


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16          # cfg.TRAINER.CLIP_AGIQA.N_CTX
        ctx_init = ""       # cfg.TRAINER.CLIP_AGIQA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224    # cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        cond = False        # cfg.TRAINER.CLIP_AGIQA.CSC

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cond:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end' # cfg.TRAINER.CLIP_AGIQA.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

from modules.clip_model import load
class CLIP_AIGCIQA(nn.Module):
    def __init__(self, classnames, use_clipagiqa_full_prompt=1):
        super().__init__()
        # self.clip = load_clip_to_cpu()
        self.clip, _ = clip.load("ViT-B/16", 'cpu', jit=False)
        self.prompt_learner = PromptLearner(classnames, self.clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip.visual
        self.text_encoder = TextEncoder(self.clip)
        self.logit_scale = self.clip.logit_scale
        self.dtype = self.clip.dtype
        # ****************************************
        # self.fc1 = nn.Linear(6 * 512, 1)
        self.regression = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*(use_clipagiqa_full_prompt+1), 512),
            # nn.Linear(512, 512),  # only images
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1),
        )
        # ****************************************

    def forward(self, image):
        image_features, _ = self.image_encoder(image.type(self.dtype))  # clip_atten中的clip/ViT返回了[bs,50,512]
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.to(torch.float32)
        text_features = text_features.to(torch.float32)

        logits = []
        for img_feat in image_features:
            # import pdb;pdb.set_trace()
            img_feat = img_feat.unsqueeze(0)
            features = torch.concat((img_feat, text_features), dim=0)
            features = features.flatten()
            logit = self.regression(features)
            logits.append(logit)

        logits = torch.stack(logits).squeeze(1)
        return logits, text_features, image_features


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

class GRM_aigciqa2(nn.Module):
    def __init__(self, device, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7, eta=1.2):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      # 去除首位，beta间距一致，保持单峰且依次递减
        self.D_grm = D_grm
        self.eta = eta
        if use_clipagiqa_full_prompt==1:
            classnames = [""]
        elif use_clipagiqa_full_prompt==6:
            classnames = ["terrible", "bad", "poor", "average","good", 'prefect']
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)

        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt
        self.txt_proj = MLP2(input_dim=512)
        self.img_proj = MLP2(input_dim=512, out_dim=1)
        self.hardsigmoid = TemperaturedTeLU_2()

    def forward(self, x, text, require_beta=False):
        batch_size = x.size(0)
        num_patch = x.size(1)
        I_resized = x[:,-1,:,:,:]

        ability, text_features, image_features = self.model(I_resized)   # [use_clipagiqa_full_prompt,512]                          
        
        params = self.img_proj(image_features)     # [bs, 1]
        if self.use_clipagiqa_full_prompt > 1:
            text_features = text_features.mean(dim=0, keepdim=True)   # [1,512]
        txt_params = self.txt_proj(text_features)     # [use_clipagiqa_full_prompt, 2]

        tp1 = tp2 = params[:, 0]        # [bs ,1]
        start = self.hardsigmoid(txt_params[:, 0], temperature=tp1) 
        # delta = self.hardsigmoid(txt_params[:, 1], temperature=tp2) + 1.2           # 2025.08.14
        delta = self.hardsigmoid(txt_params[:, 1], temperature=tp2) + self.eta
        beta = start.unsqueeze(1) + (torch.arange(self.prompt_n, device=self.device)).float() * delta.unsqueeze(1)
        if torch.any(torch.isnan(start)) or torch.any(torch.isnan(delta)) or torch.any(torch.isnan(beta)):
            import pdb;pdb.set_trace()

        logits_k = grm_sigm(ability, beta=beta, D=self.D_grm)
        if require_beta:
             return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        return self.model.clip

class GRM_aigciqa2_our_prompt(nn.Module):
    def __init__(self, device, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      # 去除首位，beta间距一致，保持单峰且依次递减
        self.D_grm = D_grm
        
        if use_clipagiqa_full_prompt==1:
            classnames = [""]
        elif use_clipagiqa_full_prompt==6:
            classnames = ["terrible", "bad", "poor", "average","good", 'prefect']
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)

        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt

    def forward(self, x, text, require_beta=False):
        batch_size = x.size(0)
        num_patch = x.size(1)
        I_resized = x[:,-1,:,:,:]

        logits_k, text_features, image_features = self.model(I_resized)   # [use_clipagiqa_full_prompt,512]                          

        ability = logits_k
        beta = logits_k.unsqueeze(1).repeat(1, self.prompt_n)
        # logits_k从[bs], 重复复制扩展未[bs, prompt_n+1]，每行元素相同
        logits_k = logits_k.unsqueeze(1).repeat(1, self.prompt_n+1)

        if require_beta:
             return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        return self.model.clip

class GRM_aigciqa2_Learnable(nn.Module):
    def __init__(self, device, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      # 去除首位，beta间距一致，保持单峰且依次递减
        self.D_grm = D_grm
        
        if use_clipagiqa_full_prompt==1:
            classnames = [""]
        elif use_clipagiqa_full_prompt==6:
            classnames = ["terrible", "bad", "poor", "average","good", 'prefect']
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)

        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt
        self.txt_proj = MLP2(input_dim=512, out_dim=prompt_n)
        self.img_proj = MLP2(input_dim=512, out_dim=prompt_n)
        self.hardsigmoid = TemperaturedTeLU_2()

    def forward(self, x, text, require_beta=False):
        batch_size = x.size(0)
        I_resized = x[:,-1,:,:,:]

        ability, text_features, image_features = self.model(I_resized)   # [use_clipagiqa_full_prompt,512]                          
        
        params = self.img_proj(image_features)     # [bs, 5]
        if self.use_clipagiqa_full_prompt > 1:
            text_features = text_features.mean(dim=0, keepdim=True)   # [1,512]
        txt_params = self.txt_proj(text_features)     # [1, 5]

        txt_params = txt_params.repeat(batch_size, 1)   # [bs, 5]
        beta = self.hardsigmoid(txt_params, temperature=params)  # [bs, prompt_n]
        # 在beta的第二个维度排序，使得它是递增的
        beta, _ = torch.sort(beta, dim=1)

        logits_k = grm_sigm(ability, beta=beta, D=self.D_grm)
        if require_beta:
             return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        return self.model.clip

class GRM_aigciqa2_geometric(nn.Module):
    def __init__(self, device, K=1, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      # 去除首位，beta间距一致，保持单峰且依次递减
        self.D_grm = D_grm
        
        if use_clipagiqa_full_prompt==1:
            classnames = [""]
        elif use_clipagiqa_full_prompt==6:
            classnames = ["terrible", "bad", "poor", "average","good", 'prefect']
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)

        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt
        self.txt_proj = MLP2(input_dim=512, out_dim=1)
        self.img_proj = MLP2(input_dim=512, out_dim=1)
        self.hardsigmoid = TemperaturedTeLU_2()
        self.monotonic_K = K

    def forward(self, x, text, require_beta=False):
        batch_size = x.size(0)
        I_resized = x[:,-1,:,:,:]

        ability, text_features, image_features = self.model(I_resized)   # [use_clipagiqa_full_prompt,512]                          
        
        params = self.img_proj(image_features)   
        if self.use_clipagiqa_full_prompt > 1:
            text_features = text_features.mean(dim=0, keepdim=True)   # [1,512]
        txt_params = self.txt_proj(text_features)    

        start = self.hardsigmoid(txt_params.squeeze(1), temperature=params.squeeze(1)) + 1.2
        # 构建公比为 K 的等比数列
        beta = start.unsqueeze(1) * (self.monotonic_K ** torch.arange(self.prompt_n, device=self.device).float().unsqueeze(0))
        logits_k = grm_sigm(ability, beta=beta, D=self.D_grm)
        if require_beta:
             return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        return self.model.clip

# ----------------------
# 1. MLP Regression Baseline (Reviewer Required)
# ----------------------
class aigciqa2_MLP_Reg(nn.Module):
    def __init__(self, device, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n
        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt

        # Same prompt initialization as your GRM (ensure fair comparison)
        if use_clipagiqa_full_prompt == 1:
            classnames = [""]
        elif use_clipagiqa_full_prompt == 6:
            classnames = ["terrible", "bad", "poor", "average", "good", 'perfect']  # Fix typo "prefect"
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)
        # Simple MLP regression head (NO GRM logic: no beta thresholds, no ability/difficulty)
        # Input: concatenated image + text features (same as GRM's feature source)
        self.regression_head = nn.Sequential(
            nn.Linear(512 + 512, 512),  # img_feat (512) + text_feat (512)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Directly regress quality score (no grade probabilities)
        )

    def forward(self, x, text, require_beta=False):
        batch_size = x.size(0)
        I_resized = x[:, -1, :, :, :]  # Same image input as GRM
        # Get image/text features (same as your GRM module)
        ability, text_features, image_features = self.model(I_resized)
        # Average text features (same as GRM when use_clipagiqa_full_prompt >1)
        if self.use_clipagiqa_full_prompt > 1:
            text_features = text_features.mean(dim=0, keepdim=True)  # [1,512]
        # Repeat text features to match batch size (critical for batching)
        text_features = text_features.repeat(batch_size, 1)  # [bs, 512]
        # Concatenate image + text features (input to MLP regression)
        concat_feat = torch.cat([image_features, text_features], dim=1)  # [bs, 1024]
        # Direct regression (no GRM sigmoid/ beta logic)
        quality_score = self.regression_head(concat_feat)  # [bs, 1]
        # Return same format as GRM for easy integration into training code
        # logits_k的shape为[bs, prompt_n+1]，每行元素相同，表示每个等级的概率（虽然没有实际意义，但保持接口一致）
        # logits_k在dim=1维度求平均，效果等价于直接复制quality_score到每个等级上，因为每行元素相同，所以平均后仍然是quality_score的值
        logits_k = quality_score.repeat(1, self.prompt_n+1)
        beta = logits_k

        if require_beta:
             return logits_k, ability, beta
        return logits_k, quality_score.squeeze(1)

    def get_clip(self):
        return self.model.clip

class GRM_aigciqa2_softmax(nn.Module):
    def __init__(self, device, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      # 去除首位，beta间距一致，保持单峰且依次递减
        self.D_grm = D_grm
        if use_clipagiqa_full_prompt==1:
            classnames = [""]
        elif use_clipagiqa_full_prompt==6:
            classnames = ["terrible", "bad", "poor", "average","good", 'prefect']
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)
        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt
        self.txt_proj = MLP2(input_dim=512, out_dim=self.prompt_n)
        self.img_proj = MLP2(input_dim=512, out_dim=self.prompt_n)
        self.hardsigmoid = TemperaturedTeLU_2()
        self.fixed_base = 1

    def forward(self, x, text, require_beta=False):
        I_resized = x[:,-1,:,:,:]

        ability, text_features, image_features = self.model(I_resized)   # [use_clipagiqa_full_prompt,512]                          
        params = self.img_proj(image_features)   
        if self.use_clipagiqa_full_prompt > 1:
            text_features = text_features.mean(dim=0, keepdim=True)   # [1,512]
        txt_params = self.txt_proj(text_features)    

        logits = self.hardsigmoid(txt_params, temperature=params) + 1.2
        # 使用softmax将logits转换为概率分布，并计算累积概率
        cum_prob = torch.cumsum(F.softmax(logits, dim=1),  dim=1)
        beta = cum_prob * self.fixed_base
        logits_k = grm_sigm(ability, beta=beta, D=self.D_grm)
        if require_beta:
             return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        return self.model.clip

from torch.distributions import Normal
class GRM_aigciqa2_CDF(nn.Module):
    def __init__(self, device, prompt_n=4, use_clipagiqa_full_prompt=1, D_grm=1.7):
        super().__init__()
        self.device = device
        self.prompt_n = prompt_n      # 去除首位，beta间距一致，保持单峰且依次递减
        self.D_grm = D_grm
        if use_clipagiqa_full_prompt==1:
            classnames = [""]
        elif use_clipagiqa_full_prompt==6:
            classnames = ["terrible", "bad", "poor", "average","good", 'prefect']
        self.model = CLIP_AIGCIQA(classnames, use_clipagiqa_full_prompt)
        self.use_clipagiqa_full_prompt = use_clipagiqa_full_prompt
        self.txt_proj = MLP2(input_dim=512)
        self.img_proj = MLP2(input_dim=512, out_dim=1)
        self.hardsigmoid = TemperaturedTeLU_2()

    def forward(self, x, text, require_beta=False):
        I_resized = x[:,-1,:,:,:]

        ability, text_features, image_features = self.model(I_resized)   # [use_clipagiqa_full_prompt,512]                          
        
        params = self.img_proj(image_features)     # [bs, 1]
        if self.use_clipagiqa_full_prompt > 1:
            text_features = text_features.mean(dim=0, keepdim=True)   # [1,512]
        txt_params = self.txt_proj(text_features)     # [use_clipagiqa_full_prompt, 2]

        tp1 = tp2 = params[:, 0]      # [bs ,1]
        mu = self.hardsigmoid(txt_params[:, 0], temperature=tp1)
        sigma = self.hardsigmoid(txt_params[:, 1], temperature=tp2) + 1.2
        mu = mu.unsqueeze(1)       # [16] → [16, 1]
        sigma = sigma.unsqueeze(1) # [16] → [16, 1]
        q = torch.linspace(0.1, 0.9, self.prompt_n, device=self.device).unsqueeze(0)  # [1, prompt_n]
        gaussian_dist = Normal(mu, sigma)
        beta = gaussian_dist.icdf(q)  # [batch_size, prompt_n], strictly increasing

        logits_k = grm_sigm(ability, beta=beta, D=self.D_grm)
        if require_beta:
             return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        return self.model.clip