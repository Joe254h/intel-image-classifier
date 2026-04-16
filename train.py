import argparse
import os
import random
from contextlib import nullcontext


CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Intel image classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pytorch", "tensorflow"],
        help="Framework to train.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory containing seg_train/seg_train and seg_test/seg_test.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default epoch count for the selected framework.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the default batch size for the selected framework.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers for PyTorch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_defaults(args):
    if args.model == "pytorch":
        epochs = args.epochs if args.epochs is not None else 70
        batch_size = args.batch_size if args.batch_size is not None else 128
    else:
        epochs = args.epochs if args.epochs is not None else 50
        batch_size = args.batch_size if args.batch_size is not None else 64
    return epochs, batch_size


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def validate_dataset_dirs(train_dir, test_dir):
    missing = [path for path in (train_dir, test_dir) if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError(
            "Missing dataset directories:\n" + "\n".join(missing)
        )


def train_pytorch(args, train_dir, test_dir, epochs, batch_size):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    from torch.amp import GradScaler, autocast

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("\n==============================================================")
    print("Training framework : PyTorch")
    print(f"Device             : {device}")
    print(f"Epochs             : {epochs}")
    print(f"Batch size         : {batch_size}")
    print(f"Learning rate      : {args.lr}")
    print(f"Train directory    : {train_dir}")
    print(f"Test directory     : {test_dir}")
    print("==============================================================\n")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_base = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_base = datasets.ImageFolder(train_dir, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transforms)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    val_size = int(0.15 * len(indices))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": args.workers,
        "pin_memory": use_amp,
        "persistent_workers": args.workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(f"Train samples      : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")
    print(f"Test samples       : {len(test_dataset)}\n")

    class JoelCNNPyTorch(nn.Module):
        def __init__(self, num_classes=6):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.15),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
            )
            self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.gap(x)
            return self.classifier(x)

    model = JoelCNNPyTorch(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=use_amp)
    amp_context = (lambda: autocast("cuda")) if use_amp else nullcontext

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    patience = 15

    print(
        f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
        f"{'Val Loss':>10} {'Val Acc':>9} {'LR':>10}"
    )
    print("-" * 65)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)

            with amp_context():
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=use_amp)
                labels = labels.to(device, non_blocking=use_amp)

                with amp_context():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_loss_avg = train_loss / max(len(train_loader), 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch + 1:>6} {train_loss_avg:>11.4f} {train_acc:>9.2f}% "
            f"{val_loss_avg:>10.4f} {val_acc:>8.2f}% {current_lr:>10.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "joel_model.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best validation accuracy: {best_val_acc:.2f}% "
                    f"at epoch {best_epoch}"
                )
                break

    print(f"\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    model.load_state_dict(torch.load("joel_model.pth", map_location=device))
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * test_correct / max(test_total, 1)
    print(f"Test accuracy      : {test_acc:.2f}%")
    print("Saved model        : joel_model.pth")


def train_tensorflow(args, train_dir, test_dir, epochs, batch_size):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models

    set_seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    strategy = (
        tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    )

    print("\n==============================================================")
    print("Training framework : TensorFlow")
    print(f"TensorFlow version  : {tf.__version__}")
    print(f"Detected GPU(s)     : {len(gpus)}")
    print(f"Distribution mode   : {type(strategy).__name__}")
    print(f"Epochs              : {epochs}")
    print(f"Batch size          : {batch_size}")
    print(f"Learning rate       : {args.lr}")
    print(f"Train directory     : {train_dir}")
    print(f"Test directory      : {test_dir}")
    print("==============================================================\n")

    autotune = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        validation_split=0.15,
        subset="training",
        seed=args.seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        validation_split=0.15,
        subset="validation",
        seed=args.seed,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    def augment(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        return image, label

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_ds = train_ds.map(augment, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(normalize, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(normalize, num_parallel_calls=autotune).prefetch(autotune)

    with strategy.scope():
        model = models.Sequential(
            [
                layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D(2, 2),
                layers.Dropout(0.25),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D(2, 2),
                layers.Dropout(0.25),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.4),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(len(CLASSES), activation="softmax"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    best_val_acc = max(history.history.get("val_accuracy", [0.0])) * 100.0

    model.save("joel_model.keras")

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy           : {test_acc * 100.0:.2f}%")
    print("Saved model             : joel_model.keras")


def main():
    args = parse_args()
    epochs, batch_size = resolve_defaults(args)

    train_dir = os.path.join(args.data_dir, "seg_train", "seg_train")
    test_dir = os.path.join(args.data_dir, "seg_test", "seg_test")
    validate_dataset_dirs(train_dir, test_dir)

    if args.model == "pytorch":
        train_pytorch(args, train_dir, test_dir, epochs, batch_size)
    else:
        train_tensorflow(args, train_dir, test_dir, epochs, batch_size)


if __name__ == "__main__":
    main()
