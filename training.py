import imageio
import pickle
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import custom_logger


logger = custom_logger("Training loop")


def train_gan(
    dataloader: DataLoader,
    generator,
    discriminator,
    device,
    gen_optimizer,
    disc_optimizer,
    loss_function,
    noise_dim: int,
    n_epochs: int,
    fixed_noise,
) -> None:
    """Function for handling training of both the Generator and Discriminator networks"""

    SAVE_RESULTS_PATH = "results/MNIST_DCGAN"

    num_iter = 0
    train_hist = {
        "D_losses": [],
        "G_losses": [],
        "epochs_times": [],
        "total_training_time": [],
    }

    logger.info("Starting training!")
    start_time = time.time()
    for epoch in range(n_epochs):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        for real_images, _ in tqdm(dataloader):

            # Train Discriminator (D)
            discriminator.zero_grad()

            # Real images
            logger.debug(f"Real images size: {real_images.size()[0]}")
            mini_batch = real_images.size()[0]
            real_images = real_images.to(device)
            y_real = torch.ones(mini_batch, device=device)

            D_result = discriminator(real_images).squeeze()
            D_real_loss = loss_function(D_result, y_real)

            # Fake images
            fake_images = (
                torch.randn((mini_batch, 100), device=device)
                .view(-1, noise_dim, 1, 1)
                .detach()
            )
            y_fake = torch.zeros(mini_batch, device=device)
            G_result = generator(fake_images)
            logger.debug(f"Generator images shape: {G_result.shape}")

            D_result = discriminator(G_result)  # .squeeze()
            logger.debug(f"Discriminator images shape: {D_result.shape}")
            D_fake_loss = loss_function(D_result, y_fake)

            # Loss calculations and backprop
            D_train_loss = D_real_loss + D_fake_loss
            D_losses.append(D_train_loss.item())

            D_train_loss.backward()
            disc_optimizer.step()

            # Train Generator (G)
            generator.zero_grad()

            # Fake images
            fake_images = torch.randn((mini_batch, 100), device=device).view(
                -1, noise_dim, 1, 1
            )

            G_result = generator(fake_images)
            D_result = discriminator(G_result).squeeze()

            # Loss calculations and backprop
            G_train_loss = loss_function(D_result, y_real)
            G_losses.append(G_train_loss.item())

            G_train_loss.backward()
            gen_optimizer.step()

            num_iter += 1

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        logger.info(
            "[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f"
            % (
                (epoch + 1),
                n_epochs,
                epoch_time,
                torch.mean(torch.FloatTensor(D_losses)),
                torch.mean(torch.FloatTensor(G_losses)),
            )
        )
        # p = 'MNIST_DCGAN_results/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        # fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        # show_result((epoch+1), save=True, path=p, isFix=False)
        # show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist["D_losses"].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist["G_losses"].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist["epochs_times"].append(epoch_time)

    end_time = time.time()
    total_training_time = end_time - start_time
    train_hist["total_training_time"].append(total_training_time)

    print(
        "Avg time per epoch: %.2f, total %d total training time: %.2f"
        % (
            torch.mean(torch.FloatTensor(train_hist["total_training_time"])),
            n_epochs,
            total_training_time,
        )
    )
    print("Training finish!... save training results")
    torch.save(generator.state_dict(), f"{SAVE_RESULTS_PATH}/generator_param.pkl")
    torch.save(
        discriminator.state_dict(), f"{SAVE_RESULTS_PATH}/discriminator_param.pkl"
    )
    with open(f"{SAVE_RESULTS_PATH}/train_hist.pkl", "wb") as f:
        pickle.dump(train_hist, f)

    # show_train_hist(train_hist, save=True, path='MNIST_DCGAN_results/MNIST_DCGAN_train_hist.png')

    # images = []
    # for e in range(n_epochs):
    #    img_name = f'{SAVE_RESULTS_PATH}/fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
    #    images.append(imageio.imread(img_name))
    # imageio.mimsave(f'{SAVE_RESULTS_PATH}/generation_animation.gif', images, fps=5)
