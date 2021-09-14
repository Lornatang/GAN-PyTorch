# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ============================================================================
# File description: Realize the model training function.
# ============================================================================
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import *


def train(dataloader, epoch) -> None:
    """Training generative models and adversarial models.

    Args:
        dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
        epoch (int): number of training cycles.
    """
    # Calculate how many iterations there are under epoch.
    batches = len(dataloader)
    # Set two models in training mode.
    discriminator.train()
    generator.train()

    for index, (real, _) in enumerate(dataloader):
        # Copy the data to the specified device.
        real = real.to(device)
        label_size = real.size(0)
        # Create label. Set the real sample label to 1, and the fake sample label to 0.
        real_label = torch.full([label_size, 1], 1.0, dtype=real.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=real.dtype, device=device)
        # Create an image that conforms to the Gaussian distribution.
        noise = torch.randn([label_size, 100], device=device)

        # Initialize the discriminator model gradient.
        discriminator.zero_grad()
        # Calculate the loss of the discriminator model on the real image.
        output = discriminator(real)
        d_loss_real = criterion(output, real_label)
        d_loss_real.backward()
        d_real = output.mean().item()
        # Generate a fake image.
        fake = generator(noise)
        # Calculate the loss of the discriminator model on the fake image.
        output = discriminator(fake.detach())
        d_loss_fake = criterion(output, fake_label)
        d_loss_fake.backward()
        d_fake1 = output.mean().item()
        # Update the weights of the discriminator model.
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.step()

        # Initialize the generator model gradient.
        generator.zero_grad()
        # Calculate the loss of the discriminator model on the fake image.
        output = discriminator(fake)
        # Adversarial loss.
        g_loss = criterion(output, real_label)
        # Update the weights of the generator model.
        g_loss.backward()
        g_optimizer.step()
        d_fake2 = output.mean().item()

        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/D_Real", d_real, iters)
        writer.add_scalar("Train_Adversarial/D_Fake1", d_fake1, iters)
        writer.add_scalar("Train_Adversarial/D_Fake2", d_fake2, iters)
        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train stage: adversarial "
                  f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
                  f"D(Real): {d_real:.6f} D(Fake1)/D(Fake2): {d_fake1:.6f}/{d_fake2:.6f}.")


def main() -> None:
    # Create a experiment result folder.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Create an image that conforms to the Gaussian distribution.
    fixed_noise = torch.randn([batch_size, 100], device=device)

    # Load dataset.
    dataset = torchvision.datasets.MNIST(root=dataset_dir,
                                         train=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, ], [0.5, ])]),
                                         download=True)
    dataloader = DataLoader(dataset, batch_size, True, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example, the power is
    # cut off in the middle of the training.
    if resume:
        print("Resuming...")
        if resume_d_weight != "" and resume_g_weight != "":
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))

    for epoch in range(start_epoch, epochs):
        # Train each epoch to generate a model.
        train(dataloader, epoch)
        # Save the weight of the model under epoch.
        torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))

        # Each epoch validates the model once.
        with torch.no_grad():
            # Switch model to eval mode.
            generator.eval()
            fake = generator(fixed_noise).detach()
            torchvision.utils.save_image(fake, os.path.join(exp_dir1, f"epoch_{epoch + 1}.bmp"), normalize=True)

    # Save the weight of the model under the last Epoch in this stage.
    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))


if __name__ == "__main__":
    main()
