# Originally made by VQGAN-CLIP (https://github.com/nerdyrodent/VQGAN-CLIP)
from icecream import ic
import csv
import argparse
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os
sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False

from torch_optimizer import DiffGrad, AdamP, RAdam

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

torch.set_num_threads(4)
default_image_size = 512
if not torch.cuda.is_available():
    default_image_size = 256
elif get_device_properties(0).total_memory <= 2 ** 33:
    default_image_size = 318

vq_parser = argparse.ArgumentParser(description='Imagination Generation with VQGAN+CLIP')
vq_parser.add_argument("-replaceable",  "--replaceable", action='store_true', help="replaceable", dest='replaceable')
vq_parser.add_argument("-nlu_dataset",  "--nlu_dataset", type=str, default="glue")
vq_parser.add_argument("-extract_textual",  "--extract_textual", action='store_true', help="eval", dest='eval')
vq_parser.add_argument('-rp', default="prompt")
vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default="test prompt", dest='prompts')
vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')
vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=100, dest='display_freq')
vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default=None, dest='init_noise')
vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','nrupdated','updatedpooling','latest'], default='latest', dest='cut_method')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'], default='Adam', dest='optimiser')
vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default="output.png", dest='output')
vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')
vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments (latest vut method only)", default=[], dest='augments')
vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')


# Execute the parse_args() method
args = vq_parser.parse_args()

data_root = "../data/nlu/"
ima_data_root = "../data/ima_gan/"
nlu_dataset = args.nlu_dataset
nli_data_path = os.path.join(data_root, nlu_dataset)
save_data_path = os.path.join(ima_data_root, nlu_dataset)
if not os.path.exists(save_data_path): os.makedirs(save_data_path)

if not args.augments:
   args.augments = [['Af', 'Pe', 'Ji', 'Er']]

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

# Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
# NB. May not work for AMD cards?
if not args.cuda_device == 'cpu' and not torch.cuda.is_available():
    args.cuda_device = 'cpu'
    print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
    print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")


# NR: Testing with different intital images
def random_noise_image(w,h):
    random_image = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def random_gradient_image(w,h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


#NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow # not used with pooling
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        
        for _ in range(self.cutn):            
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


# Generate Imagination
device = torch.device(args.cuda_device)
model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
jit = False
perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

def iterate_func():
    cut_size = perceptor.visual.input_resolution
    f = 2**(model.decoder.num_resolutions - 1)

    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)

    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f

    if gumbel:
        e_dim = 256
        n_toks = model.quantize.n_embed
        z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


    if args.init_image:
        if 'http' in args.init_image:
            img = Image.open(urlopen(args.init_image))
        else:
            img = Image.open(args.init_image)
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == 'pixels':
        img = random_noise_image(args.size[0], args.size[1])    
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == 'gradient':
        img = random_gradient_image(args.size[0], args.size[1])
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        # z = one_hot @ model.quantize.embedding.weight
        if gumbel:
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
        #z = torch.rand_like(z)*2						# NR: check

    z_orig = z.clone()
    z.requires_grad_(True)

    pMs = []
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    # CLIP tokenize/encode   
    if args.prompts:
        for prompt in args.prompts:
            txt, weight, stop = split_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = split_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))


    # Set the optimiser
    def get_opt(opt_name, opt_lr):
        if opt_name == "Adam":
            opt = optim.Adam([z], lr=opt_lr)	# LR=0.1 (Default)
        elif opt_name == "AdamW":
            opt = optim.AdamW([z], lr=opt_lr)	
        elif opt_name == "Adagrad":
            opt = optim.Adagrad([z], lr=opt_lr)	
        elif opt_name == "Adamax":
            opt = optim.Adamax([z], lr=opt_lr)	
        elif opt_name == "DiffGrad":
            opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
        elif opt_name == "AdamP":
            opt = AdamP([z], lr=opt_lr)		    
        elif opt_name == "RAdam":
            opt = RAdam([z], lr=opt_lr)		    
        elif opt_name == "RMSprop":
            opt = optim.RMSprop([z], lr=opt_lr)
        else:
            print("Unknown optimiser. Are choices broken?")
            opt = optim.Adam([z], lr=opt_lr)
        return opt

    opt = get_opt(args.optimiser, args.step_size)


    # Output for the user
    print('Using device:', device)
    print('Optimising using:', args.optimiser)

    if args.prompts:
        print('Using text prompts:', args.prompts)  
    if args.image_prompts:
        print('Using image prompts:', args.image_prompts)
    if args.init_image:
        print('Using initial image:', args.init_image)
    if args.noise_prompt_weights:
        print('Noise prompt weights:', args.noise_prompt_weights)    


    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed  
    torch.manual_seed(seed)
    print('Using seed:', seed)


    # Vector quantize
    def synth(z):
        if gumbel:
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


    #@torch.no_grad()
    @torch.inference_mode()
    def checkin(i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        # tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)
        info = PngImagePlugin.PngInfo()
        info.add_text('comment', f'{args.prompts}')
        TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info) 	


    def ascend_txt():
        global i
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
        
        result = []

        if args.init_weight:
            # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*args.init_weight) / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        return result # return loss


    def train(i):
        opt.zero_grad(set_to_none=True)
        lossAll = ascend_txt()
        
        if i % args.display_freq == 0:
            checkin(i, lossAll)
        
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        
        #with torch.no_grad():
        with torch.inference_mode():
            z.copy_(z.maximum(z_min).minimum(z_max))


    i = 0 # Iteration idx+1er
    p = 1 # Phrase idx+1er
    try:
        with tqdm() as pbar:
            while True:            
                # Change text prompt
                if args.prompt_frequency > 0:
                    if i % args.prompt_frequency == 0 and i > 0:
                        # In case there aren't enough phrases, just loop
                        if p >= len(all_phrases):
                            p = 0
                        
                        pMs = []
                        args.prompts = all_phrases[p]

                        # Show user we're changing prompt                                
                        print(args.prompts)
                        
                        for prompt in args.prompts:
                            txt, weight, stop = split_prompt(prompt)
                            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                            pMs.append(Prompt(embed, weight, stop).to(device))
         
                        p += 1

                # Training time
                train(i)
                if i == args.max_iterations: break
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass


def task_iter():
    if os.path.exists(args.output) and not args.replaceable:
        return
    if args.prompts:
        story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]
        all_phrases = []
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))
        args.prompts = all_phrases[0]
    iterate_func()
        

if __name__ == '__main__':
        
    registered_path = {
        'glue_cola_train': os.path.join(nli_data_path, "CoLA/train.tsv"),
        'glue_cola_dev': os.path.join(nli_data_path, "CoLA/dev.tsv"),
        'glue_cola_test': os.path.join(nli_data_path, "CoLA/test.tsv"),

        'glue_mnli_train': os.path.join(nli_data_path, "MNLI/train.tsv"),
        'glue_mnli_m_dev': os.path.join(nli_data_path, "MNLI/dev_matched.tsv"),
        'glue_mnli_mm_dev': os.path.join(nli_data_path, "MNLI/dev_mismatched.tsv"),
        'glue_mnli_m_test': os.path.join(nli_data_path, "MNLI/test_matched.tsv"),
        'glue_mnli_mm_test': os.path.join(nli_data_path, "MNLI/test_mismatched.tsv"),

        'glue_mrpc_train': os.path.join(nli_data_path, "MRPC/train.tsv"),
        'glue_mrpc_dev': os.path.join(nli_data_path, "MRPC/dev.tsv"),
        'glue_mrpc_test': os.path.join(nli_data_path, "MRPC/test.tsv"),

        'glue_sst_train': os.path.join(nli_data_path, "SST-2/train.tsv"),
        'glue_sst_dev': os.path.join(nli_data_path, "SST-2/dev.tsv"),
        'glue_sst_test': os.path.join(nli_data_path, "SST-2/test.tsv"),

        'glue_sts_train': os.path.join(nli_data_path, "STS-B/train.tsv"),
        'glue_sts_dev': os.path.join(nli_data_path, "STS-B/dev.tsv"),
        'glue_sts_test': os.path.join(nli_data_path, "STS-B/test.tsv"),

        'glue_rte_train': os.path.join(nli_data_path, "RTE/train.tsv"),
        'glue_rte_dev': os.path.join(nli_data_path, "RTE/dev.tsv"),
        'glue_rte_test': os.path.join(nli_data_path, "RTE/test.tsv"),

        'glue_wnli_train': os.path.join(nli_data_path, "WNLI/train.tsv"),
        'glue_wnli_dev': os.path.join(nli_data_path, "WNLI/dev.tsv"),
        'glue_wnli_test': os.path.join(nli_data_path, "WNLI/test.tsv"),

        'glue_qnli_train': os.path.join(nli_data_path, "QNLI/train.tsv"),
        'glue_qnli_dev': os.path.join(nli_data_path, "QNLI/dev.tsv"),
        'glue_qnli_test': os.path.join(nli_data_path, "QNLI/test.tsv"),

        'glue_qqp_train': os.path.join(nli_data_path, "QQP/train.tsv"),
        'glue_qqp_dev': os.path.join(nli_data_path, "QQP/dev.tsv"),
        'glue_qqp_test': os.path.join(nli_data_path, "QQP/test.tsv"),
    }
    dev_list = [args.prompts]
    if args.rp in registered_path.keys():
        raw_filepath = registered_path[args.rp]
        output_path = os.path.join(save_data_path, "output", args.rp)
        if not os.path.isdir(output_path): os.makedirs(output_path)

        tsv_file = open(raw_filepath)
        dev_list = csv.reader(tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE)

    for idx, line in enumerate(dev_list):
        if "mnli" in args.rp:
            label = line[-1] if not "test" in args.rp else "test"
            args.prompts = line[-8 if "dev" in args.rp else -4].replace("\"","").replace(":","")
            args.output = "%s/%s_sentence1.png"%(output_path,idx+1)
            task_iter()
            args.prompts = line[-7 if "dev" in args.rp else -3].replace("\"","").replace(":","")   
            args.output = "%s/%s_sentence2.png"%(output_path,idx+1)
            task_iter()  
        elif "qnli" in args.rp or "wnli" in args.rp or "rte" in args.rp:
            label = line[-1] if not "test" in args.rp else "test"
            args.prompts = line[1].replace("\"","").replace(":","")
            args.output = "%s/%s_sentence1.png"%(output_path,idx+1)
            task_iter()
            args.prompts = line[2].replace("\"","").replace(":","")   
            args.output = "%s/%s_sentence2.png"%(output_path,idx+1)
            task_iter()  
        elif "sst" in args.rp:
            args.prompts = line[0].replace("\"","").replace(":","")   
            label = line[1] if not "test" in args.rp else "test"
            args.output = "%s/%s_%s.png"%(output_path,idx+1,label)
            task_iter()
        elif "sts" in args.rp:
            try:
                label = line[9] if not "test" in args.rp else "test"
            except:
                ic(line)
            args.prompts = line[7].replace("\"","").replace(":","")   
            args.output = "%s/%s_sentence1.png"%(output_path,idx+1)
            task_iter()
            args.prompts = line[8].replace("\"","").replace(":","")   
            args.output = "%s/%s_sentence2.png"%(output_path,idx+1)
            task_iter()
        elif "cola" in args.rp:
            label = line[1] if not "test" in args.rp else "test"
            args.prompts = line[-1].replace("\"","").replace(":","")   
            args.output = "%s/%s_%s.png"%(output_path,idx+1,label)
            task_iter()
        elif "mrpc" in args.rp:
            label = line[0] if not "test" in args.rp else "test"
            args.prompts = line[-2].replace("\"","").replace(":","")
            sid = line[-4]
            args.output = "%s/%s_sentence1.png"%(output_path,idx+1)
            task_iter()
            args.prompts = line[-1].replace("\"","").replace(":","")   
            sid = line[-3]
            args.output = "%s/%s_sentence2.png"%(output_path,idx+1)
            task_iter()
        elif "qqp" in args.rp:
            label = line[-1] if not "test" in args.rp else "test"
            args.prompts = line[3] if not "test" in args.rp else line[1]
            args.prompts = args.prompts.replace("\"","").replace(":","")
            args.output = "%s/%s_sentence1.png"%(output_path,idx+1)
            task_iter()
            args.prompts = line[4] if not "test" in args.rp else line[2]
            args.prompts = args.prompts.replace("\"","").replace(":","")
            args.output = "%s/%s_sentence2.png"%(output_path,idx+1)
            task_iter()
        else:
            args.output = os.path.join(save_data_path, "prompt_test_output.png")
            task_iter()