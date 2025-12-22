"""
UIFace-Plus: 扩散模型人脸生成可视化系统

本脚本实现了基于扩散模型的人脸生成，并可视化去噪过程。
主要功能：
1. 使用 DDIM 采样生成高质量人脸
2. 集成 Classifier-Free Guidance (CFG) 提升生成质量
3. 可视化扩散模型的去噪过程（从噪声到清晰人脸）

作者：复旦大学 生成模型课程
日期：2024年12月
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from pathlib import Path

generation_dir = "E:/FDU/课程/生成模型/TFace-master/generation"
uiface_dir = os.path.join(generation_dir, "uiface")
sys.path.insert(0, uiface_dir)

# 添加父目录到路径
from models.diffusion.unet import ConditionalUNet
from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from diffusion.ddpm import DenoisingDiffusionProbabilisticModel


class UIFacePlusGenerator:
    """增强版 UIFace 生成器，支持可视化扩散过程"""

    def __init__(self, config_path, checkpoint_path, vq_encoder_path, vq_decoder_path, device='cuda'):
        self.device = device

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 加载扩散模型
        print("Loading diffusion model...")
        self.unet = self._load_unet(checkpoint_path)
        self.unet.to(device)
        self.unet.eval()

        # 加载自动编码器
        print("Loading autoencoder...")
        self.vq_encoder = self._load_autoencoder(vq_encoder_path, encoder=True)
        self.vq_decoder = self._load_autoencoder(vq_decoder_path, encoder=False)

        # 创建扩散过程
        self.diffusion = DenoisingDiffusionProbabilisticModel(
            eps_model=self.unet,
            T=1000,
            schedule_type='linear',
            schedule_beta_min=0.0001,
            schedule_beta_max=0.02
        )

        # 动态计算 latent 空间尺寸（参考官方 sample.py:56,65）
        image_size = (3, 128, 128)  # UIFace 训练时使用的图像尺寸
        with torch.no_grad():
            dummy_input = torch.ones([1, *image_size]).to(device)
            self.latent_shape = self.vq_encoder(dummy_input).shape[1:]  # (C, H, W)
        print(f"Latent shape: {self.latent_shape}")

        print("Model loaded successfully!")

    def _load_unet(self, checkpoint_path):
        """加载 UNet 模型"""
        # 使用 UIFace 的配置参数
        unet = ConditionalUNet(
            input_channels=3,
            initial_channels=96,
            channel_multipliers=(1, 2, 2, 2),
            is_attention=(False, True, True, True),
            attention_heads=-1,
            attention_head_channels=32,
            n_blocks_per_resolution=2,
            condition_type="CA",
            is_context_conditional=True,
            n_context_classes=0,
            context_input_channels=512,
            context_channels=256,
            learn_empty_context=True,
            context_dropout_probability=0.25,
            unconditioned_probability=0.2,
        )

        # 加载权重
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 提取实际的 state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 去掉 'module.eps_model.' 前缀，并过滤掉 diffusion 相关的参数
            diffusion_keys = {'betas', 'alphas', 'sigmas', 'sqrt_alphas_inv', 'alpha_bars',
                             'sqrt_alpha_bars', 'sqrt_one_minus_alpha_bars',
                             'one_minus_alphas_over_sqrt_one_minus_alpha_bars',
                             'alphas_prev', 'alphas_next'}

            new_state_dict = {}
            for key, value in state_dict.items():
                # 跳过 diffusion 相关的参数
                if any(key.endswith(dk) or key == f'module.{dk}' for dk in diffusion_keys):
                    continue

                if key.startswith('module.eps_model.'):
                    new_key = key.replace('module.eps_model.', '')
                    new_state_dict[new_key] = value
                elif key.startswith('module.'):
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            unet.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")

        return unet

    def _load_autoencoder(self, path, encoder=True):
        """加载自动编码器"""
        if not os.path.exists(path):
            return None

        # VQ-VAE 需要配置文件路径
        config_path = os.path.join(uiface_dir, 'models', 'autoencoder', 'first_stage_config.yaml')

        if encoder:
            model = VQEncoderInterface(config_path, path)
        else:
            model = VQDecoderInterface(config_path, path)

        model.to(self.device)
        model.eval()
        return model

    def _get_betas(self):
        """获取扩散过程的 beta schedule"""
        num_steps = self.config.get('diffusion', {}).get('num_diffusion_timesteps', 1000)
        beta_schedule = self.config.get('diffusion', {}).get('beta_schedule', 'linear')

        if beta_schedule == 'linear':
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
        else:
            raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")

    @torch.no_grad()
    def generate_with_steps(self, identity_embedding=None, num_steps=50, save_intermediate=True, cfg_scale=1.5):
        """
        生成人脸并保存中间步骤

        Args:
            identity_embedding: 身份嵌入向量，如果为 None 则随机生成
            num_steps: DDIM 采样步数
            save_intermediate: 是否保存中间步骤
            cfg_scale: Classifier-Free Guidance 强度（参考官方 sample.py）

        Returns:
            final_image: 最终生成的图像
            intermediate_images: 中间步骤的图像列表
        """
        # 初始化噪声（在 latent 空间）
        latent_shape = (1, *self.latent_shape)  # 使用动态计算的 latent 尺寸
        x_t = torch.randn(latent_shape, device=self.device)

        # 如果没有提供身份嵌入，使用随机嵌入并归一化（参考官方 sample.py:85-86）
        if identity_embedding is None:
            identity_embedding = np.random.randn(512)
            identity_embedding = identity_embedding / np.linalg.norm(identity_embedding)
            identity_embedding = torch.tensor(identity_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            identity_embedding = torch.tensor(identity_embedding, device=self.device).unsqueeze(0)

        # 采样步骤（参考官方 ddpm.py:82）
        skip = self.diffusion.T // num_steps

        intermediate_images = []
        intermediate_latents = []

        print(f"Generating face with {num_steps} denoising steps (skip={skip}, CFG scale={cfg_scale})...")

        # DDIM 采样循环（参考官方 ddpm.py:87-115）
        for i in reversed(range(0, self.diffusion.T, skip)):
            t_batch = torch.tensor([i], device=self.device)

            # CFG: 预测无条件和有条件的噪声（参考官方 ddpm.py:228-244）
            noise_pred_uncond, _, _ = self.unet(x_t, t_batch, None)  # 无条件
            noise_pred_cond, _, _ = self.unet(x_t, t_batch, identity_embedding)  # 有条件

            # CFG 组合（参考官方 ddpm.py:242-244）
            noise_pred = (1 + cfg_scale) * noise_pred_cond - cfg_scale * noise_pred_uncond

            # DDIM 更新（参考官方 ddpm.py:94-115）
            prev_timestep = i - skip
            alpha_prod_t = self.diffusion.alpha_bars[i]
            alpha_prod_t_prev = (
                self.diffusion.alphas_prev[prev_timestep]
                if prev_timestep >= 0
                else torch.tensor(1.0, device=self.device)
            )
            beta_prod_t = 1 - alpha_prod_t

            # 预测 x0（参考官方 ddpm.py:101-102）
            pred_x0 = (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

            # 计算 x_{t-1}（参考官方 ddpm.py:106-109，eta=0的DDIM）
            pred_x0_direction = torch.sqrt(alpha_prod_t_prev) * pred_x0
            pred_noise_direction = torch.sqrt(1 - alpha_prod_t_prev) * noise_pred
            x_t = pred_x0_direction + pred_noise_direction

            # 保存中间步骤
            if save_intermediate and (i % (self.diffusion.T // 10) == 0 or i == 0):
                intermediate_latents.append(x_t.cpu().clone())

        # 解码最终潜在表示
        if self.vq_decoder is not None:
            final_image = self.vq_decoder(x_t)  # forward 方法就是解码
            final_image = self._tensor_to_image(final_image)

            # 解码中间步骤
            for latent in intermediate_latents:
                img = self.vq_decoder(latent.to(self.device))
                intermediate_images.append(self._tensor_to_image(img))
        else:
            final_image = self._tensor_to_image(x_t)
            intermediate_images = [self._tensor_to_image(lat.to(self.device)) for lat in intermediate_latents]

        return final_image, intermediate_images

    def _tensor_to_image(self, tensor):
        """将 tensor 转换为 PIL Image"""
        # 反归一化
        img = tensor.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def visualize_denoising_process(self, intermediate_images, save_path):
        """可视化去噪过程"""
        num_images = len(intermediate_images)
        fig, axes = plt.subplots(2, (num_images + 1) // 2, figsize=(15, 6))
        axes = axes.flatten()

        for i, img in enumerate(intermediate_images):
            axes[i].imshow(img)
            axes[i].set_title(f'Step {i * (100 // num_images)}%')
            axes[i].axis('off')

        # 隐藏多余的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Denoising visualization saved to {save_path}")


def main():
    """主函数"""
    # 配置路径（需要用户填写）
    config = {
        'model_config': './generation/uiface/configs/model/unet_cond_ca_cpd25_uncond20.yaml',
        'checkpoint': 'UIFace-Plus\models\\ema_averaged_model_250000.ckpt',  # 需要下载
        'vq_encoder': 'UIFace-Plus\models\\first_stage_encoder_state_dict.pt',  # 需要下载
        'vq_decoder': 'UIFace-Plus\models\\first_stage_decoder_state_dict.pt',  # 需要下载
    }

    # 检查文件是否存在
    for key, path in config.items():
        if not os.path.exists(path) and 'path/to' not in path:
            print(f"Warning: {key} not found at {path}")

    # 创建生成器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        generator = UIFacePlusGenerator(
            config_path=config['model_config'],
            checkpoint_path=config['checkpoint'],
            vq_encoder_path=config['vq_encoder'],
            vq_decoder_path=config['vq_decoder'],
            device=device
        )

        # 创建输出目录
        output_dir = Path('outputs/generation')
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成多个人脸
        num_faces = 5
        for i in range(num_faces):
            print(f"\n=== Generating face {i+1}/{num_faces} ===")

            # 生成人脸（使用官方推荐的50步DDIM采样）
            final_image, intermediate_images = generator.generate_with_steps(
                num_steps=100,
                save_intermediate=True,
                cfg_scale=1.5  # CFG引导强度，官方默认值
            )

            # 保存最终图像
            final_image.save(output_dir / f'face_{i+1}_final.png')
            print(f"Saved final image to {output_dir / f'face_{i+1}_final.png'}")

            # 可视化去噪过程
            if intermediate_images:
                generator.visualize_denoising_process(
                    intermediate_images,
                    output_dir / f'face_{i+1}_denoising_process.png'
                )

        print("\n✅ Generation completed successfully!")
        print(f"📁 Check outputs in: {output_dir.absolute()}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 提示: 请先下载预训练模型权重:")
        print("   1. UIFace 扩散模型: https://drive.google.com/drive/folders/11OnYj0mtEkepjl3gE2oLeDJu_WeuB0Ma")
        print("   2. VQ-VAE 编码器/解码器: https://drive.google.com/drive/folders/1d-zs3yjsnzOMNkz7qy3JSb-fMf0UmSdT")
        print("\n   然后更新配置文件中的路径")


if __name__ == '__main__':
    main()
