import pytorch_lightning as pl
import torch
import wandb
import os


class SlotAttentionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=8, save_path="./", name="example"):
        super().__init__()
        print(val_samples)
        self.val_samples = val_samples['image'][:num_samples]
        print(f'Shape of val_samples: {self.val_samples.shape}\n')
        self.save_path = save_path
        self.name = name

    def on_validation_epoch_end(self, trainer, pl_module):
        val_samples = self.val_samples.to(device=pl_module.device)
        result, recons, _= pl_module(val_samples)
        trainer.logger.experiment.log({
            'images': [wandb.Image(x/2 + 0.5) for x in torch.clamp(val_samples, -1, 1)],
            'reconstructions': [wandb.Image(x/2 + 0.5) for x in torch.clamp(result, -1, 1)]
        })
        trainer.logger.experiment.log({
            f'{i} slot': [wandb.Image(x/2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]
             for i in range(pl_module.num_slots)
        })
        torch.save(pl_module.state_dict(), os.path.join(self.save_path, f"{self.name}.pth"))
