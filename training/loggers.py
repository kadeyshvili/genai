import wandb
import torch
from PIL import Image
from collections import defaultdict
import os

class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip())
        # if config.train.checkpoint_path != "":
        #     resume_path = self.checkpoint_dir / config.trainer.resume_from
        #     self._resume_checkpoint(resume_path)
        #     raise NotImplementedError()
        # else:

        wandb.init(
            project=config.logger.project_name,
            entity=config.logger.entity,
            name=config.logger.run_name,
            resume="allow",  # resume the run if run_id existed
            mode=config.logger.mode,
        )
        self.wandb = wandb



    @staticmethod
    def log_values(self, values_dict: dict, step: int):
        self.wandb.log(
            {
                scalar_name: scalar
                for scalar_name, scalar in values_dict.items()
            },
            step=step,
        )

    @staticmethod
    def log_images(self, images: dict, step: int):
        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image) for image_name, image in images.items()}, step=step
        )

class TrainingLogger:
    def __init__(self, config):
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)


    def log_train_losses(self, step: int):
        averaged_losses = {loss_name: sum(values) / len(values) for loss_name, values in self.losses_memory.items()}
        
        self.logger.log_values(averaged_losses, step)
        
        self.losses_memory.clear()


    def log_val_metrics(self, val_metrics: dict, step: int):
        self.logger.log_values(val_metrics, step)



    def log_batch_of_images(self, batch: torch.Tensor, step: int, images_type: str = ""):
        images = {f"{images_type}_{idx}": image for idx, image in enumerate(batch)}

        self.logger.log_images(images, step)



    def update_losses(self, losses_dict):
            # It is useful to average losses over a number of steps rather than track them at each step.
            # This makes the training curves smoother.
            for loss_name, loss_val in losses_dict.items():
                self.losses_memory[loss_name].append(loss_val)





