from utils.class_registry import ClassRegistry 
from training.trainers.base_trainer import BaseTrainer 
from models.diffusion_models import diffusion_models_registry 
from training.optimizers import optimizers_registry 
from training.losses.diffusion_losses import DiffusionLossBuilder 
import torch
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import os
from torchvision.utils import save_image

 
diffusion_trainers_registry = ClassRegistry() 
 
@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer") 
class BaseDiffusionTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

    def setup_models(self):
        # model_config = self.config.model
        # self.unet = diffusion_models_registry.get(model_config.name)(**model_config.params)
        self.model = UNet2DModel(
            sample_size=128,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        if self.config.train.checkpoint_path:
            checkpoint = torch.load(self.config.checkpoint_path)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            self.noise_scheduler.load_state_dict(checkpoint['noise_scheduler_state_dict'])

    def setup_optimizers(self):
        optimizer_config = self.config.train.optimizer
        self.optimizer = optimizers_registry[optimizer_config](
            self.model.parameters())
        
        if self.config.train.checkpoint_path:
            checkpoint = torch.load(self.config.checkpoint_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def setup_losses(self):
        self.loss_builder = DiffusionLossBuilder(self.config)

    def to_train(self):
        self.model.train()

    def to_eval(self):
        self.model.eval()

    def train_step(self):
        self.to_train()
        batch = next(self.train_dataloader)
        images = batch['images']
        noise = torch.rand(images.shape).to(images.device)
        bs = images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=images.device).long()
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        loss = self.loss_builder.calculate_loss(noise_pred, noise)[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'noise_scheduler_state_dict': self.noise_scheduler.state_dict(),
        }
        torch.save(checkpoint, f'{self.config.experiment_dir}/checkpoint_{epoch}.pth')

    def synthesize_images(self, num_images=16):
        self.to_eval()
        generated_images = []
        sample = torch.randn(8, 3, 128, 128).to(self.device)
        for i, t in enumerate(self.noise_scheduler.timesteps):
            with torch.no_grad():
                residual = self.model(sample, t).sample
            sample = self.noise_scheduler.step(residual, t, sample).prev_sample
        generated_images.append(sample)
        
        save_path = f'{self.config.to_save.experiment_dir}/images'
        os.makedirs(save_path, exist_ok=True)
        for idx, img in enumerate(generated_images):
            save_image(img, f'{save_path}/image_{idx}.png')
        
        return torch.stack(generated_images), save_path