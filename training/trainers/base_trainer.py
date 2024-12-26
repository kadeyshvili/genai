import torch 
 
from abc import abstractmethod 
from datasets.dataloaders import InfiniteLoader 
from training.loggers import TrainingLogger 
from datasets.datasets import datasets_registry 
from metrics.metrics import metrics_registry 
import os 
from tqdm import tqdm 
from torchvision import transforms

 
 
class BaseTrainer: 
    def __init__(self, config): 
        self.config = config 
 
        self.device = config.exp.device 
        self.start_step = config.train.start_step 
        self.step = 0 
     
 
    def setup(self): 
        self.setup_experiment_dir() 
 
        self.setup_models() 
        self.setup_optimizers() 
        self.setup_losses() 
 
        self.setup_metrics() 
        self.setup_logger() 
 
        self.setup_datasets() 
        self.setup_dataloaders() 
 
 
    def setup_inference(self): 
        self.setup_experiment_dir() 
 
        self.setup_models() 
 
        self.setup_metrics() 
        self.setup_logger() 
 
        self.setup_datasets() 
        self.setup_dataloaders() 
 
 
    @abstractmethod 
    def setup_models(self): 
        pass 
 
    @abstractmethod 
    def setup_optimizers(self): 
        pass 
 
    @abstractmethod 
    def setup_losses(self): 
        pass 
 
    @abstractmethod 
    def to_train(self): 
        pass 
 
    @abstractmethod 
    def to_eval(self): 
        pass 
 
    def setup_experiment_dir(self): 
        experiment_name = self.config.exp.name 
        self.experiment_dir = os.path.join(self.config.exp.base_dir, experiment_name) 
        os.makedirs(self.experiment_dir, exist_ok=True) 
 
 
    def setup_metrics(self): 
        self.metrics = [] 
        for metric_name in self.config.train.val_metrics: 
            metric = metrics_registry[metric_name](test_pth=self.config.data.input_val_dir) 
            # metric = metrics_registry[metric_name](test_pth='/home/andrewut/food_data/test/apple_pie') 
            
            self.metrics.append(metric) 

 
    def setup_logger(self): 
        self.logger = TrainingLogger(self.config) 
 
 
    def setup_datasets(self): 
        dataset_name = self.config.data.name 
        self.train_dataset = datasets_registry[dataset_name](self.config.data.input_train_dir) 
        self.val_dataset = datasets_registry[dataset_name](self.config.data.input_val_dir) 
 
    def setup_dataloaders(self): 
        batch_size_train = self.config.data.train_batch_size 
        self.train_dataloader = InfiniteLoader(self.train_dataset, batch_size=batch_size_train, shuffle=True) 
        batch_size_val = self.config.data.val_batch_size 
 
        self.val_dataloader = InfiniteLoader(self.val_dataset, batch_size=batch_size_val, shuffle=True) 
 
 
    def training_loop(self): 
        self.to_train() 
 
        for self.step in tqdm(range(self.start_step, self.config.train.steps + 1)): 
            losses_dict = self.train_step() 
            self.logger.update_losses(losses_dict) 
 
            if self.step % self.config.train.val_step == 0: 
                val_metrics_dict, images = self.validate() 
                print("FID:", val_metrics_dict.values())
                self.logger.log_val_metrics({"learning rate":self.lr_scheduler.get_last_lr()[0]}, step=self.step)
                self.logger.log_val_metrics(val_metrics_dict, step=self.step) 
                self.logger.log_batch_of_images(images, step=self.step, images_type="validation") 
 
            if self.step % self.config.train.log_step == 0: 
                self.logger.log_train_losses(self.step) 
 
            if self.step % self.config.train.checkpoint_step == 0: 
                self.save_checkpoint(self.step) 
 
 
    @abstractmethod 
    def train_step(self): 
        pass 
 
    @abstractmethod 
    def save_checkpoint(self): 
        pass 
 
 

 
 
 
    @torch.no_grad() 
    def validate(self): 
        self.to_eval() 
        images_sample, images_pth = self.synthesize_images() 
 
        metrics_dict = {} 
        for metric in self.metrics: 
            metrics_dict[metric] = metric( 
                synt_pth=images_pth 
            ) 
        return metrics_dict, images_sample 
 
 
    @abstractmethod 
    def synthesize_images(self): 
        pass 
 
 
    @torch.no_grad() 
    def inference(self): 
        # TO DO 
        # Validate your model, save images 
        # Calculate metrics 
        # Log if needed 
        raise NotImplementedError() 
 
 

