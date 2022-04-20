import os

import flash
import torch
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy

seed_everything(42)


def main():
    data_path = f"{os.getcwd()}/data"

    download_data(
        "https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", f"{data_path}/"
    )

    datamodule = ImageClassificationData.from_folders(
        train_folder=f"{data_path}/hymenoptera_data/train/",
        val_folder=f"{data_path}/hymenoptera_data/val/",
        test_folder=f"{data_path}/hymenoptera_data/test/",
        batch_size=4,
        transform_kwargs={
            "image_size": (196, 196),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
    )

    model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

    trainer = flash.Trainer(
        max_epochs=100,
        gpus=torch.cuda.device_count(),
        # accelerator="gpu",
        # devices=torch.cuda.device_count(),
        # strategy=DDPStrategy(find_unused_parameters=False),
    )

    trainer.finetune(model, datamodule=datamodule, strategy="freeze")


if __name__ == "__main__":
    main()
