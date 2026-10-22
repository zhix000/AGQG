from torch.utils.data import DataLoader
from ImageDataset import  AIGCDataset,AIGCDataset_3k,AIGCIQA2023Dataset

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

from PIL import Image
import logging

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def set_dataset_aigc(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):

    data = AIGCDataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader

def set_dataset_aigc_3k(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):

    data = AIGCDataset_3k(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_dataset_aigc_2023(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):

    data = AIGCIQA2023Dataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader



class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation #插值方式，默认为双线性插值
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    #调用一个对象时即使用圆括号时，Python会自动查找并执行该对象的__call__方法。
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return transforms.Resize(self.size, self.interpolation)(img)
        else:
            return img


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512), #调整图像大小到512 
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess4():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess5():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)  

#--------------------------------------------------------------------------
from torch.utils.data import Sampler

class GroupedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.data_source.reset_index(drop=True, inplace=True)  # 重置索引从0开始，并丢弃旧的索引

    def __iter__(self):
        grouped_data = self.data_source.groupby(1)  # 按照prompt分组
        groups = grouped_data.groups  # 获取每个组的索引列表
        for label, indices in groups.items():
            for idx in indices:
                yield idx  # 生成索引

    def __len__(self):
        return len(self.data_source)

def set_dataset_aigc_3k_align(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):
    data = AIGCDataset_3k(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind)

    grouped_sampler = GroupedSampler(data.data)
    loader = DataLoader(data, batch_size=bs, sampler=grouped_sampler, pin_memory=True, num_workers=num_workers)

    return loader

def check_loader(data_loader):
    print('the length of data_loader is ', len(data_loader))
    for i, batch in enumerate(data_loader):
        prompts = batch['prompt']
        image_names = batch['image_name']

        print(f"Batch {i} prompt: {prompts}")
        print(f"Batch {i} image names: {image_names}")
        if i == 1:  # Print only the first batch to avoid spamming the output
            break

def set_dataset_iqa2023_align(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, blind=False):
    data = AIGCIQA2023Dataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess,
        blind=blind)

    grouped_sampler = GroupedSampler(data.data)
    loader = DataLoader(data, batch_size=bs, sampler=grouped_sampler, pin_memory=True, num_workers=num_workers)

    return loader