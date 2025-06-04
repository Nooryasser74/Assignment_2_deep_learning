import torch
import wandb

class WandBLogger:
    def __init__(self, enabled=True, 
                 model: torch.nn.Module = None, 
                 run_name: str = None) -> None:
        self.enabled = enabled

        if self.enabled:
            wandb.init(
                entity="noorlasheen135-tu-wien",        
                project="segformer-task5",              # ✅ Changed for Task 5
                group="SegFormer_Task5_Comparison",     # ✅ Changed group name
                name=run_name if run_name else None     
            )

            if model is not None:
                self.watch(model)

    def watch(self, model, log_freq: int = 1):
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step is not None:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def finish(self):
        if self.enabled:
            wandb.finish()
