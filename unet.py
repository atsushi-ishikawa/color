import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import os


def create_hint_image(color_image_tensor, prob=0.01):
    if color_image_tensor.dim() == 4:
        color_image_tensor = color_image_tensor[0]

    _, h, w = color_image_tensor.shape
    mask = torch.rand((h, w)) < prob
    mask = mask.float().unsqueeze(0).repeat(3, 1, 1).to(color_image_tensor.device)
    hint = color_image_tensor * mask

    if hint.sum() < 0.1:
        noise = tourch.rand_like(hint) * mask
        hint = hint + 0.2 * noise
        hint = torch.clamp(hint, 0.0, 1.0)

    return hint


class LineArtDataset(Dataset):
    def __init__(self, color_dir, line_dir, transform=None, max_samples=None):
        self.color_paths = sorted([os.path.join(color_dir, f) for f in os.listdir(color_dir)])
        self.line_paths  = sorted([os.path.join(line_dir, f) for f in os.listdir(line_dir)])

        if max_samples is not None:
            self.color_paths = self.color_paths[:max_samples]
            self.line_paths = self.line_paths[:max_samples]

        self.transform = transform

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        color = Image.open(self.color_paths[idx]).convert("RGB")
        line = Image.open(self.line_paths[idx]).convert("L")

        if self.transform:
            color = self.transform(color)
            line = self.transform(line)

        gamma = 0.8
        color = color ** gamma
        line = line.repeat(3, 1, 1)

        return line, color


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = UNetBlock(6, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dec3 = UNetBlock(512 + 256, 256)
        self.dec2 = UNetBlock(256 + 128, 128)
        self.dec1 = UNetBlock(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

        nn.init.xavier_uniform_(self.final.weight)
        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, 0.0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # return torch.sigmoid(self.final(d1))
        return self.final(d1)
        # return torch.tanh(self.final(d1))*0.5 + 0.5  # soft clipping


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:23].eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.layer_weights = layer_weights or [1.0] * 4

    def to(self, device):
        self.vgg = self.vgg.to(device)
        return super().to(device)

    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        layer_ids = [3, 8, 15, 22]

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in layer_ids:
                idx = layer_ids.index(i)
                loss += self.layer_weights[idx] * self.criterion(x, y)

        return loss


def normalize_batch(batch):
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


def colorfulness_loss(output):
    r, g, b = output[:, 0], output[:, 1], output[:, 2]
    return -((r - g).abs().mean() + (r - b).abs().mean())


def apply_bilateral_filter(img_path, save_path):
    img = cv2.imread(img_path)
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(save_path, filtered)


def upscale_image(img_path, save_path, scale=2):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_path, upscaled)


transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = LineArtDataset("anime_dataset/color", "anime_dataset/line", transform=transform, max_samples=60)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)

# criterion = nn.L1Loss()
l1_loss = nn.L1Loss()
perc_loss = PerceptualLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

os.makedirs("outputs", exist_ok=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (line, color) in enumerate(loader):
        line = line.to(device)
        color = color.to(device)

        zero_hint = random.random() < 0.0

        if zero_hint:
            hint = torch.zeros_like(line)
        else:
            hint = torch.stack([create_hint_image(c) for c in color]).to(device)

        input_tensor = torch.cat([line, hint], dim=1)

        optimizer.zero_grad()
        output = model(input_tensor)

        if zero_hint:
            loss = (l1_loss(output, color) + 0.06 * colorfulness_loss(output)
                                           + 0.01 * perc_loss(normalize_batch(output), normalize_batch(color)))
        else:
            loss = (l1_loss(output, color) + 0.03 * colorfulness_loss(output)
                                           + 0.01 * perc_loss(normalize_batch(output), normalize_batch(color)))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i == 0:
            output_clamped = torch.clamp(output[0], 0.0, 1.0)
            line = input_tensor[0][:3].detach().cpu()
            hint = input_tensor[0][3:].detach().cpu()

            save_path = f"outputs/pred_epoch_{epoch:02}.png"
            # filtered_path = f"outputs/pred_epoch_{epoch:02}_filtered.png"
            # final_path = f"outputs/pred_epoch_{epoch:02}_final.png"

            save_image(output_clamped, save_path)
            save_image(color[0], f"outputs/gt_epoch_{epoch:02}.png")
            save_image(line, f"outputs/input_epoch_{epoch:02}.png")
            save_image(hint, f"outputs/hint_epoch_{epoch:02}.png")

            # apply_bilateral_filter(save_path, filtered_path)
            # upscale_image(filtered_path, final_path, scale=2)

            print(f"Output min: {output.min().item():.3f}, max: {output.max().item():.3f}")
            r, g, b = output[0][0], output[0][1], output[0][2]
            print(f"R-G diff: {(r - g).abs().mean().item():.2f}")
            print(f"R-B diff: {(r - b).abs().mean().item():.2f}")

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(loader):.3f}")

line_img = Image.open("my_lineart.png").convert("L")
line_tensor = transform(line_img).unsqueeze(0)
line_tensor = line_tensor.repeat(1, 3, 1, 1)

hint_tensor = torch.zeros_like(line_tensor)

input_tensor = torch.cat([line_tensor, hint_tensor], dim=1).to(device)

model.eval()
with torch.no_grad():
    output = model(input_tensor)
    output_clamped = torch.clamp(output[0], 0.0, 1.0)
    save_image(output_clamped, "outputs/my_result.png")
