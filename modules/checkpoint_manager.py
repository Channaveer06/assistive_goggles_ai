import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_ROOT = os.path.join(BASE_DIR, "project_data", "checkpoints")

def train_yolo_with_auto_resume(data_yaml, run_name="custom_assistive_yolo", total_epochs=50, imgsz=640, batch=16):
    """
    Saves a checkpoint after EVERY epoch (save_period=1) on Google Drive.
    If Colab disconnects mid-training, re-running this function automatically resumes from last.pt.
    """
    last_ckpt_path = os.path.join(CKPT_ROOT, run_name, "weights", "last.pt")

    if os.path.exists(last_ckpt_path):
        print(f"🔄 Disconnect recovery triggered! Resuming training from {last_ckpt_path}...")
        model = YOLO(last_ckpt_path)
        return model.train(resume=True)
    else:
        print(f"🚀 Starting fresh training for '{run_name}' with every-epoch Drive backups...")
        base_pt = os.path.join(BASE_DIR, "project_data", "models_onnx", "yolo11n.pt")
        root_pt = os.path.join(BASE_DIR, "yolo11n.pt")
        model = YOLO(base_pt if os.path.exists(base_pt) else (root_pt if os.path.exists(root_pt) else "yolo11n.pt"))
        return model.train(
            data=data_yaml,
            epochs=total_epochs,
            imgsz=imgsz,
            batch=batch,
            project=CKPT_ROOT,
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=1,  # Saves checkpoint to Google Drive after EVERY single epoch
        )

class PyTorchEpochCheckpointer:
    """
    Universal per-epoch saver & auto-resumer for any custom PyTorch model (e.g., currency/door classifier).
    Preserves model weights, optimizer state, learning rate scheduler, and exact epoch number on Drive.
    """
    def __init__(self, experiment_name):
        self.save_dir = os.path.join(CKPT_ROOT, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.latest_path = os.path.join(self.save_dir, "latest_epoch.pt")
        self.best_path = os.path.join(self.save_dir, "best_model.pt")
        self.best_loss = float("inf")

    def load_if_exists(self, model, optimizer=None, scheduler=None, device="cuda"):
        """Call before starting your epoch loop. Returns the next epoch index to run."""
        if not os.path.exists(self.latest_path):
            print("🆕 No previous checkpoint found on Drive. Starting from Epoch 0.")
            return 0

        ckpt = torch.load(self.latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.best_loss = ckpt.get("best_loss", float("inf"))
        next_epoch = ckpt["epoch"] + 1
        print(f"⚡ Restored checkpoint from Google Drive! Resuming cleanly at Epoch {next_epoch} (Best Loss: {self.best_loss:.4f})")
        return next_epoch

    def save_epoch(self, epoch, model, optimizer, val_loss, scheduler=None):
        """Call at the end of every single epoch to flush progress to Google Drive."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "best_loss": min(self.best_loss, val_loss),
        }
        # Atomic write via temp file so a disconnect during saving never corrupts latest_epoch.pt
        temp_path = self.latest_path + ".tmp"
        torch.save(state, temp_path)
        os.replace(temp_path, self.latest_path)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(state, self.best_path)
            print(f"💾 [Epoch {epoch}] Saved to Drive + New Best Model (Loss: {val_loss:.4f})")
        else:
            print(f"💾 [Epoch {epoch}] Checkpoint synced to Google Drive (Loss: {val_loss:.4f})")
