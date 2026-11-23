"""Official compositional splits, exact metadata filenames and seeded augmentation."""
from itertools import product
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

SPLIT = 'compositional-split-natural'


def image_transform(training=False):
    operations = [T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)]
    if training:
        operations += [T.RandomHorizontalFlip(), T.RandomPerspective(), T.RandomRotation(5)]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


class CompositionData:
    def __init__(self, root):
        self.root = Path(root)
        for phase in ('train', 'val', 'test'):
            pairs = [tuple(line.split()) for line in
                     (self.root / SPLIT / f'{phase}_pairs.txt').read_text().splitlines() if line.strip()]
            if not pairs or any(len(p) != 2 for p in pairs) or len(set(pairs)) != len(pairs):
                raise ValueError(f'Invalid {phase} pair list')
            setattr(self, phase + '_pairs', pairs)
        self.pairs = sorted(set(self.train_pairs + self.val_pairs + self.test_pairs))
        self.attrs = sorted({a for a, _ in self.pairs})
        self.objs = sorted({o for _, o in self.pairs})
        self.open_pairs = list(product(self.attrs, self.objs))
        self.records = {phase: [] for phase in ('train', 'val', 'test')}
        permitted = set(self.pairs)
        metadata = torch.load(self.root / f'metadata_{SPLIT}.t7', map_location='cpu', weights_only=True)
        for row in metadata:
            a, o, phase = row['attr'], row['obj'], row['set']
            if a == 'NA' or phase == 'NA' or (a, o) not in permitted:
                continue
            if phase not in self.records:
                raise ValueError(f'Unknown metadata phase: {phase}')
            self.records[phase].append((row['image'], a, o))
        groups = [{row[0] for row in self.records[p]} for p in self.records]
        if any(len(g) != len(self.records[p]) for g, p in zip(groups, self.records)):
            raise ValueError('Duplicate image identity within a split')
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Image identity overlap across splits')
        train_set = set(self.train_pairs)
        if any((a, o) not in train_set for _, a, o in self.records['train']):
            raise ValueError('Training record outside seen compositions')

    def candidates(self, setting):
        return self.open_pairs if setting == 'open' else self.pairs

    def mask(self, pairs, phase, setting):
        allowed = set(self.train_pairs + getattr(self, phase + '_pairs'))
        return torch.tensor([setting == 'open' or pair in allowed for pair in pairs])


class Images(Dataset):
    def __init__(self, data, phase, pairs, *, augment=False, epoch=1, seed=42):
        self.data, self.records = data, data.records[phase]
        self.lookup = {p: i for i, p in enumerate(pairs)}
        self.transform, self.augment = image_transform(augment), augment
        self.seed, self.epoch = seed, epoch
        if augment and phase != 'train':
            raise ValueError('Augmentation is restricted to training images')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name, a, o = self.records[index]
        # Use the filename in metadata verbatim, including MIT underscore folders.
        with Image.open(self.data.root / 'images' / name) as image:
            image = image.convert('RGB')
            if self.augment:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(self.seed * 1000003 + self.epoch * 9176 + index)
                    pixels = self.transform(image)
            else:
                pixels = self.transform(image)
        return pixels, self.lookup[(a, o)]
