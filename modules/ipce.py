import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the CLIP module is available in your environment
# from clip_atten import clip 
import clip

def compute_grm_probabilities(theta: torch.Tensor, beta: torch.Tensor, alpha: float = 1.0, D: float = 1.702) -> torch.Tensor:
    """
    Compute the category probabilities using the Graded Response Model (GRM).
    This is a fully vectorized implementation for efficiency.
    
    Math:
        P*_k(theta) = sigmoid(D * alpha * (theta - beta_k))
        P_1 = 1 - P*_1
        P_k = P*_{k-1} - P*_k  (for 1 < k < K)
        P_K = P*_{K-1}
        
    Args:
        theta: Image ability scores, shape [batch_size].
        beta: Difficulty thresholds, shape [batch_size, num_thresholds]. 
              Note: num_thresholds = num_categories - 1.
        alpha: Discrimination parameter.
        D: Scaling constant.
        
    Returns:
        probabilities: Category probabilities, shape [batch_size, num_categories].
    """
    batch_size = theta.size(0)
    device = theta.device
    
    # 1. Compute cumulative probabilities P*_k = sigmoid(D * alpha * (theta - beta_k))
    # theta: [B, 1], beta: [B, K-1] -> cum_probs: [B, K-1]
    cum_probs = torch.sigmoid(D * alpha * (theta.unsqueeze(1) - beta))
    
    # 2. Pad with P*_0 = 1 and P*_K = 0 to facilitate subtraction
    # padded_cum_probs: [B, K+1]
    pad_left = torch.ones(batch_size, 1, device=device, dtype=cum_probs.dtype)
    pad_right = torch.zeros(batch_size, 1, device=device, dtype=cum_probs.dtype)
    padded_cum_probs = torch.cat([pad_left, cum_probs, pad_right], dim=1)
    
    # 3. Compute category probabilities P_k = P*_{k-1} - P*_k
    # probabilities: [B, K]
    probabilities = padded_cum_probs[:, :-1] - padded_cum_probs[:, 1:]
    
    return probabilities


class FeatureProjector(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for projecting CLIP features 
    into latent ability or difficulty parameters.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, 
                 num_layers: int = 2, out_dim: int = 2):
        super().__init__()
        
        layers = []
        ch_in = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(ch_in, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            # Use LeakyReLU for hidden layers, Tanh for the last hidden layer
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                layers.append(nn.Tanh())
            ch_in = hidden_dim
            
        layers.append(nn.Linear(ch_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemperaturedTeLU(nn.Module):
    """
    Temperatured TeLU activation function.
    Math: f(x, T) = (x + T) * tanh(exp(x + T))
    
    Used to modulate the text-derived difficulty priors with the 
    image-dependent temperature coefficient, ensuring smooth and bounded 
    nonlinear transformation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        z = x + temperature
        return z * torch.tanh(torch.exp(z))


class AGQG_IPCE(nn.Module):
    """
    AGQG-enhanced IPCE model for AI-Generated Image Quality Assessment.
    
    This module integrates the Arithmetic Graded Response Model (AGRM) 
    into the IPCE framework, decoupling quality assessment into image ability 
    estimation and difficulty modeling.
    """
    def __init__(self, device, num_thresholds: int = 4, D_grm: float = 2.0, eta: float = 1.2):
        """
        Args:
            device: Torch device to place the model on.
            num_thresholds: Number of difficulty thresholds (K-1, where K is the number of quality grades).
                            Default is 4 (for 5 quality grades: L5).
            D_grm: Scaling constant D in the GRM formulation.
            eta: Small positive hyperparameter ensuring positivity and minimal separation 
                 between beta_1 and gamma (eta > 0).
        """
        super().__init__()
        self.device = device
        self.num_thresholds = num_thresholds     
        self.D_grm = D_grm
        self.eta = eta

        # Load CLIP backbone (ViT-B/32)
        self.clip, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.clip.logit_scale.requires_grad = False
        
        # Projection heads for ability and difficulty estimation
        self.txt_proj = FeatureProjector(input_dim=512, out_dim=2)  # Outputs beta_1^T and gamma^T
        self.img_proj = FeatureProjector(input_dim=512, out_dim=1)  # Outputs temperature coefficient tau_I

        self.activation = TemperaturedTeLU()

    def forward(self, images: torch.Tensor, texts: list, require_beta: bool = False):
        """
        Args:
            images: Input image patches, shape [batch_size, num_patches, C, H, W].
            texts: List of text prompts corresponding to each image.
            require_beta: If True, returns the estimated difficulty thresholds beta.
            
        Returns:
            logits_k: Category probabilities, shape [batch_size, num_categories].
            ability: Estimated image ability scores, shape [batch_size].
            beta: (Optional) Estimated difficulty thresholds, shape [batch_size, num_thresholds].
        """
        batch_size, num_patches = images.size(0), images.size(1)
        
        # Reshape images for CLIP encoding: [B*patch, C, H, W]
        images_flat = images.view(-1, images.size(2), images.size(3), images.size(4))

        # 1. Extract CLIP features
        image_features, _ = self.clip.encode_image(images_flat)         # [B*patch, 512]
        text_features, _, _ = self.clip.encode_text(texts, return_ori=True)   # [B, 512]
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 2. Compute Image Ability (theta)
        # Calculate cosine similarity scaled by a fixed factor (10)
        logits_per_image = 10.0 * (image_features @ text_features.t())  # [B*patch, B]
        logits_per_image = logits_per_image.view(batch_size, num_patches, -1)
        
        # IPCE-specific patch aggregation: average over spatial patches, keep the last patch separate
        logits_per_image = torch.cat((
            logits_per_image[:, :num_patches-1, :].mean(1).unsqueeze(1), 
            logits_per_image[:, -1, :].unsqueeze(1)
        ), dim=1)
        
        # Average the aggregated patches and extract the diagonal (image-text alignment)
        logits_per_image_p = logits_per_image.mean(1)
        ability = torch.diagonal(logits_per_image_p)  # [B]
        
        # 3. Compute Quality Difficulty (beta_1, gamma)
        # Average image features across patches for difficulty estimation
        image_features_avg = image_features.view(batch_size, num_patches, -1).mean(1)  # [B, 512]
        
        # Project features to latent parameters
        tau_I = self.img_proj(image_features_avg).squeeze(-1)       # [B], temperature coefficient
        txt_params = self.txt_proj(text_features)                   # [B, 2], text-derived priors
        
        beta_1_T = txt_params[:, 0]  # Text-derived base difficulty
        gamma_T = txt_params[:, 1]   # Text-derived difficulty interval
        
        # Apply Temperatured TeLU to modulate priors with image-dependent temperature
        beta_1 = self.activation(beta_1_T, temperature=tau_I)
        gamma = self.activation(gamma_T, temperature=tau_I) + self.eta  # Add eta for positivity
        
        # Construct arithmetic sequence: beta_k = beta_1 + (k-1)*gamma
        # k goes from 0 to num_thresholds-1
        arange = torch.arange(self.num_thresholds, device=self.device, dtype=beta_1.dtype)
        beta = beta_1.unsqueeze(1) + arange.unsqueeze(0) * gamma.unsqueeze(1)  # [B, num_thresholds]

        # 4. Compute Category Probabilities using GRM
        logits_k = compute_grm_probabilities(ability, beta=beta, D=self.D_grm)
        
        if require_beta:
            return logits_k, ability, beta
        return logits_k, ability

    def get_clip(self):
        """Returns the CLIP backbone for external weight conversion (e.g., FP16/FP32 handling)."""
        return self.clip