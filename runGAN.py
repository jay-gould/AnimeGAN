#imports
from gan64 import AnimeDataset, Generator64
import torch
import sys
from torchvision.transforms import GaussianBlur
sys.path.append('ESRGAN')
from ESRGANtest import run_ESRGAN

def generate_images():
    #get AnimeDataset for functions
    anime_dataset = AnimeDataset(download=False)
    #get anime generator
    generator = Generator64()
    #load trained weights
    generator.load_state_dict(torch.load('./generator64.pt'))
    #index for the generated image
    index = 0
    while True:
        choice = input('Generate New Image? (Y/N): ')

        if choice.upper() == 'Y':
            file_path = './ESRGAN/LR/gen_anime_' + str(index) + '.png'
            index += 1 
            noise_vector = torch.randn(1, 64, 1, 1, device='cpu')
            output = generator(anime_dataset.denorm(noise_vector))
            blur_kernel = GaussianBlur(3, 0.3)
            output_blurred = blur_kernel(output)
            #interpolate to smooth the image
            #output = torch.nn.functional.interpolate(output,(128,128), mode = 'bilinear')
            anime_dataset.save_image(image = output_blurred, file_name = file_path)
            run_ESRGAN('cpu')
            
        elif choice.upper() == 'N':
            break
        else:
            print('Type Y or N')

if __name__=='__main__':
    generate_images()
