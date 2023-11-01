#imports
import torch
import torch.nn as nn
import opendatasets as od
import os
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

#Generator model
class Generator64(nn.Module):
  def __init__(self):
    super(Generator64, self).__init__()
    self.main = nn.Sequential(
      #ConvTranspose2d has arguments (in_channels, out_channels, kernel_size (4x4), stride=1, padding=0, output_padding=0)
      # Block 1:input is Z, going into a convolution
      nn.ConvTranspose2d(64, 64 * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(64 * 8),
      nn.ReLU(True),
      # Block 2: input is (64 * 8) x 4 x 4
      nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 4),
      nn.ReLU(True),
      # Block 3: input is (64 * 4) x 8 x 8
      nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 2),
      nn.ReLU(True),
      # Block 4: input is (64 * 2) x 16 x 16
      nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      # Block 5: input is (64) x 32 x 32
      nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
      nn.Tanh()
      # Output: output is (3) x 64 x 64
      )
  def forward(self, input):
    output = self.main(input)
    return output

#discriminator
class Discriminator64(nn.Module):
  def __init__(self):
    super(Discriminator64,self).__init__()
    self.main = nn.Sequential(
      #Block 1: input is (3) x 64 x 64
      nn.Conv2d(3, 64, 4, 2, 1, bias = False),
      nn.LeakyReLU(0.2, inplace=True),
      #Block 2: input is (64) x 32 x 32
      nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # Block 3: input is (64*2) x 16 x 16
      nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # Block 4: input is (64*4) x 8 x 8
      nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # Block 5: input is (64*8) x 4 x 4
      nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid(),
      nn.Flatten()
      # Output: 1
    )

  def forward(self, input):
    output = self.main(input)
    return output

class AnimeDataset():
  def __init__(self, download = True):
    self.data_directory = './animefacedataset'
    self.image_size = 64
    self.batch_size = 128
    self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if download == True:
      self.download_dataset()

  def download_dataset(self):
    #download dataset
    dataset_url = 'https://www.kaggle.com/splcher/animefacedataset'
    od.download(dataset_url)
    self.data_dir = './animefacedataset'

  def create_dataloader(self):
    #transform the images and place into imagefolder
    self.train_dataset = ImageFolder(self.data_dir, transform=transforms.Compose([
    transforms.Resize(self.image_size),
    transforms.CenterCrop(self.image_size),
    transforms.ToTensor(),
    transforms.Normalize(*self.stats)]))
    #place images into dataloader
    train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return train_dataloader

  def denorm(self, img_tensors):
    return img_tensors * self.stats[1][0] + self.stats[0][0]

  def show_images(self, images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(self.denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()

  def save_image(self, image, file_name, nmax=64):
    image_denormed =self.denorm(image.detach())
    save_image(image_denormed, file_name)

  def show_batch(self, dataloader, nmax=64):
    for images, _ in dataloader:
      self.show_images(images, nmax)
      break

class TrainingProcess():
  def __init__(self):
    #generator and discriminator
    self.generator = Generator64()
    self.discriminator = Discriminator64()
    #loss function
    self.adversarial_loss = nn.BCELoss()
    #dataset and dataloader
    self.anime_dataset = AnimeDataset()
    self.train_dataloader = self.anime_dataset.create_dataloader()
    #device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #number of epochs to train across and learning rate
    self.num_epochs = 50
    self.learning_rate = 0.002
    #setup optimiser
    self.generator_optimiser = optim.Adam(self.generator.parameters(), lr = self.learning_rate, betas=(0.5, 0.999))
    self.discrim_optimiser = optim.Adam(self.discriminator.parameters(), lr = self.learning_rate, betas=(0.5, 0.999))

    #run training process


  #fake output is values discriminator get from generated (fake) images
  #label is 1's only. this is because if fake_output = 0, then
  #discriminator guessed correctly, so we need to update
  #and if = 1 then we have generated good fake image
  def generator_loss_func(self, fake_output, label):
    gen_loss = self.adversarial_loss(fake_output, label)
    return gen_loss

  #this will be called twice, once on real data and once on fake data
  def discriminator_loss(self, output, label):
    disc_loss = self.adversarial_loss(output,label)
    return disc_loss

  #training
  def training_process(self):
    #move model to device (GPU if available)
    self.generator.to(self.device)
    self.discriminator.to(self.device)
    #loop over epochs
    for epoch in range(self.num_epochs):
      discrim_loss_list, generator_loss_list = [],[]

      for index, (real_images, _) in enumerate(self.train_dataloader):
        self.discrim_optimiser.zero_grad()
        real_images = real_images.to(self.device)

        real_target = Variable(torch.ones(real_images.size(0)).to(self.device)) #all ones
        real_target = real_target.unsqueeze(1)
        fake_target = Variable(torch.zeros(real_images.size(0)).to(self.device)) #all zeros
        fake_target = fake_target.unsqueeze(1)

        #training discriminator with real images
        output = self.discriminator(real_images) #should be all ones if discriminator is correct
        discrim_real_loss = self.discriminator_loss(output, real_target)
        discrim_real_loss.backward()

        #training discriminator with fake images
        noise_vector = torch.randn(real_images.size(0), 64, 1, 1, device=self.device)
        noise_vector = noise_vector.to(self.device)
        generated_image = self.generator(noise_vector)
        output = self.discriminator(generated_image.detach()) #detach as we don't want to train generator with this information
        discrim_fake_loss = self.discriminator_loss(output,fake_target) #output should all be zeros if discriminator is correct
        discrim_fake_loss.backward()

        #calculate total loss for discriminator
        discrim_total_loss = discrim_real_loss + discrim_fake_loss
        discrim_loss_list.append(discrim_total_loss)

        #update discriminator parameters
        self.discrim_optimiser.step()

        #training generator
        self.generator_optimiser.zero_grad()
        generator_output = self.discriminator(generated_image) #do not detatch this time as we want to train generator
        generator_loss = self.generator_loss_func(generator_output, real_target) #want the discriminator to output all ones, so that is the target loss
        generator_loss_list.append(generator_loss)
        generator_loss.backward()
        self.generator_optimiser.step()

    #once trained move models back to CPU
    self.generator.to('cpu')
    self.discriminator.to('cpu')
    return discrim_loss_list, generator_loss_list, self.generator, self.discriminator

def train_and_save_GAN(colab = False):
  trainer = TrainingProcess()
  discrim_loss_list, generator_loss_list, generator, discrimintator = trainer.training_process()
  if colab == True:
    #Google Collab setup
    from google.colab import drive
    drive.mount('/content/drive/')
    data_dir = '/content/drive/My Drive/self_projects/gan64/'
  else:
    data_dir = './'
  torch.save(generator.state_dict(), data_dir + 'generator64.pt')
  torch.save(discrimintator.state_dict(), data_dir + 'discriminator64.pt')
  return discrim_loss_list, generator_loss_list, generator, discrimintator

if __name__ == '__main__':
  discrim_loss_list, generator_loss_list, generator, discrimintator = train_and_save_GAN(colab = True)
  anime_dataset = AnimeDataset(download = False)
  noise_vector = torch.randn(1, 64, 1, 1, device='cpu')
  output = generator(noise_vector)
  anime_dataset.show_images(output)