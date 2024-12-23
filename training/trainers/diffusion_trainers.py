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
        self.model = UNet2DModel(
            sample_size=self.config.model.sample_size,  # the target image resolution
            in_channels=self.config.model.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=self.config.model.out_channels,  # the number of output channels
            layers_per_block=self.config.model.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=self.config.model.block_out_channels,  # More channels -> more parameters
            down_block_types=self.config.model.down_block_types,
            up_block_types=self.config.model.up_block_types,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.config.scheduler_args.steps)

        if self.config.train.checkpoint_path:
            checkpoint = torch.load(self.config.checkpoint_path)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            self.noise_scheduler.load_state_dict(checkpoint['noise_scheduler_state_dict'])

    def setup_optimizers(self):
        optimizer_name = self.config.train.optimizer
        self.optimizer = optimizers_registry[optimizer_name](
            self.model.parameters(), self.config.optimizer_args)
        
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
        images = batch['images'].to(self.device)
        noise = torch.rand(images.shape).to(images.device)
        bs = images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=images.device).long()
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps).to(self.device)
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        loss = self.loss_builder.calculate_loss(noise_pred, noise)[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def save_checkpoint(self, step):
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{self.config.to_save.experiment_dir}/checkpoint_{step}.pth')

    def synthesize_images(self):
        self.to_eval()
        save_path = os.path.join(self.config.to_save.experiment_dir, 'generated_images')
        os.makedirs(save_path, exist_ok=True)
        generated_images = []
        
        for idx in range(25):
            sample = torch.randn(1, 3, 64, 64).to(self.device)
            for t in self.noise_scheduler.timesteps:
                with torch.no_grad():
                    residual = self.model(sample, t).sample
                sample = self.noise_scheduler.step(residual, t, sample).prev_sample
        
            sample = (sample + 1) / 2
            sample = sample.clamp(0, 1)

            save_image(sample, f'{save_path}/image_{idx}.jpg')
            generated_images.append(sample)
        generated_images = torch.cat(generated_images, dim=0)
        return generated_images, save_path
