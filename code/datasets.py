from __future__ import print_function
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class Dataset_HFM(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        #self.data = pd.read_csv(csv_file)
        data = torch.load(csv_file)
        self.examples = data['examples']
        self.energies = data['energies']
        self.root_dir = root_dir
        self.transform = transform
        #self.data['examples'] = self.data['examples'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
        #self.data['energies'] = self.data['energies'].apply(lambda x: torch.tensor(int(x), dtype=torch.int32))



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = self.examples[idx]
        energy = self.energies[idx]
        #example = self.data.iloc[idx]['examples']
        #energy = self.data.iloc[idx]['energies']
        return example, energy
    


class Dataset_pureHFM(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        #self.data = pd.read_csv(csv_file)
        data = torch.load(csv_file)
        self.examples = data['examples']
        self.energies = data['energies']
        self.root_dir = root_dir
        self.transform = transform
        #self.data['examples'] = self.data['examples'].apply(lambda x: torch.tensor(np.fromstring(x.strip("[]"), sep=' '), dtype=torch.float32))
        #self.data['energies'] = self.data['energies'].apply(lambda x: torch.tensor(int(x), dtype=torch.int32))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = self.examples[idx]
        energy = self.energies[idx]
        #example = self.data.iloc[idx]['examples']
        #energy = self.data.iloc[idx]['energies']
        return example, energy
    


# https://github.com/genyrosk/pytorch-VAE-models/blob/master/dsprites/data_dsprites.py

class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, filepath, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        dataset_zip = np.load(filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        value = self.latents_values[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)
        return sample, value


def load_dsprites(filepath='/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                train_split=0.9, shuffle=True, seed=42, batch_size=64, reduce_dataset = True):
    # img_size = 64
    # path = os.path.join(dir, 'dsprites-dataset')

    dataset = DisentangledSpritesDataset(filepath, transform=transforms.ToTensor())



    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    if reduce_dataset:
        max_train = 60000
        max_val = 10000
        train_indices = train_indices[:max_train]
        val_indices = val_indices[:max_val]
    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader




class MNISTDigit2Dataset(Dataset):
    """
    Custom dataset containing only digit '2' from MNIST, 
    augmented to 60,000 samples using symmetry transformations
    """
    
    def __init__(self, train=True, download=True, target_size=60000):
        self.target_size = target_size
        
        # Load original MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST(root='/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data', train=train, download=download, transform=transform)
        
        # Filter only digit '2' samples
        digit_2_data = []
        digit_2_targets = []
        
        for i, (image, label) in enumerate(mnist):
            if label == 2:
                digit_2_data.append(image)
                digit_2_targets.append(label)
        
        self.original_data = torch.stack(digit_2_data)
        self.original_targets = torch.tensor(digit_2_targets)
        
        print(f"Found {len(self.original_data)} original samples of digit '2'")
        
        # Generate augmented dataset
        self.augmented_data, self.augmented_targets = self._generate_augmented_data()
        
    def _generate_augmented_data(self):
        """Generate augmented data using symmetry transformations"""
        num_original = len(self.original_data)
        augmented_data = []
        augmented_targets = []
        
        # Define transformation functions
        transformations = [
            lambda x: x,  # Original
            lambda x: torch.rot90(x, k=1, dims=[-2, -1]),  # 90° rotation
            lambda x: torch.rot90(x, k=2, dims=[-2, -1]),  # 180° rotation
            lambda x: torch.rot90(x, k=3, dims=[-2, -1]),  # 270° rotation
            lambda x: torch.flip(x, dims=[-1]),  # Horizontal flip
            lambda x: torch.flip(x, dims=[-2]),  # Vertical flip
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]),  # 90° + h_flip
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2]),  # 90° + v_flip
        ]
        
        # Calculate how many times we need to replicate the data
        replications_needed = (self.target_size + num_original - 1) // num_original
        
        for rep in range(replications_needed):
            for i, (image, target) in enumerate(zip(self.original_data, self.original_targets)):
                if len(augmented_data) >= self.target_size:
                    break
                
                # Choose a transformation (cycle through them)
                transform_idx = (rep * num_original + i) % len(transformations)
                transformed_image = transformations[transform_idx](image)
                
                augmented_data.append(transformed_image)
                augmented_targets.append(target)
            
            if len(augmented_data) >= self.target_size:
                break
        
        # Trim to exact target size
        augmented_data = augmented_data[:self.target_size]
        augmented_targets = augmented_targets[:self.target_size]
        
        print(f"Generated {len(augmented_data)} augmented samples")
        
        return torch.stack(augmented_data), torch.tensor(augmented_targets)
    
    def __len__(self):
        return len(self.augmented_data)
    
    def __getitem__(self, idx):
        return self.augmented_data[idx], self.augmented_targets[idx]
