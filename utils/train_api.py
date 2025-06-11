import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
import os
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import random
from sklearn.utils.class_weight import compute_class_weight

def vis_signal(ground_truth, reconstructed, save_dir, epoch, num_channels_to_plot=5):
    # Number of channels to plot
    num_channels_to_plot = 5
    signal_length = ground_truth.shape[1]
    # Create a figure for the subplots
    fig, axes = plt.subplots(num_channels_to_plot, figsize=(15, 10))  # 5 rows, 1 column

    # Define colors for ground truth and reconstructed signals
    color_ground_truth = 'blue'
    color_reconstructed = 'red'

    # Plot the first five channels for both ground truth and reconstructed signals
    for i in range(num_channels_to_plot):
        axes[i].plot(ground_truth[i], color=color_ground_truth, label='Ground Truth')
        axes[i].plot(reconstructed[i], color=color_reconstructed, linestyle='dashed', label='Reconstructed')
        print(reconstructed[i])
        print(ground_truth[i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_xlim([0, signal_length])
        axes[i].legend()
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_dir + 'signals_epoch_{}.png'.format(epoch))
    plt.close(fig)

def eval_KNN(vae, train_data, test_data, train_label, test_label):
    vae.eval()
    neigh = KNeighborsClassifier(n_neighbors=5)

    
    train_data = torch.from_numpy(train_data).cuda().float()
    train_data_logits = vae.module.encoder(train_data)
    _, quantized_train_data, _, _, train_code_indices = vae.module._vq_vae(train_data_logits)
    quantized_train_data = quantized_train_data.permute(0, 2, 1).detach().cpu().numpy()

    test_data = torch.from_numpy(test_data).cuda().float()
    test_data_logits = vae.module.encoder(test_data)
    _, quantized_test_data, _, _, train_code_indices = vae.module._vq_vae(test_data_logits)
    quantized_test_data = quantized_test_data.permute(0, 2, 1).detach().cpu().numpy()


    neigh.fit(quantized_train_data.reshape(quantized_train_data.shape[0], -1), train_label)
    pred = neigh.predict(quantized_test_data.reshape(quantized_test_data.shape[0], -1))
    return accuracy_score(test_label, pred)

class CustomToTensor:
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic = np.array(pic).astype('float32')
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ClassRemap:
    def __call__(self, x):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        x -= 1
        return x


class Uint8Remap:
    def __call__(self, x):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        x /= 255
        return x


def image_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.copy()
        # return img


def save_model(path, vae):
    save_obj = {
        # 'hparams': vae.hparams,
        # vae_params,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)


def build_data_set(data, split='train', seed=42, mode=1):
    if(mode == 0):
        ds = SEEGDataset_multisub(
            data.data_folder,
            split,
        )
    elif(mode == 1):
        ds = PaddedSEEGDataset_multisub(
            data.data_folder,
            split,
            seed,
        )
    elif(mode == 2):
        ds = sEEGDataset_multisub_masking(
            data.data_folder,
            split,
            seed,
        )
    else:
        # Raise an error to indicate that other cases are not implemented
        raise NotImplementedError("This dataset mode has not been implemented yet.")

    return ds

# def build_data_set(data, split='all'):

#     ds = SEEGDataset(
#         data.data_folder,
#         split,
#     )
#     return ds

class SEEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', seed=42):

        self.data = torch.load(self.scandir(data_root, split, str(seed)))['samples']
        self.label = torch.load(self.scandir(data_root, split, str(seed)))['labels']

    def __getitem__(self, index):

        return self.data[index], self.data[index]
    
    def scandir(self, directory, split, seed):
        matching_files = []

        # Walk through all files in the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if both keywords are in the file name
                if split in file:
                # if split in file and seed in file:
                    file_path = os.path.join(root, file)
                    matching_files.append(file_path)
        return matching_files[0]
    def __len__(self):
        return self.data.shape[0]

class SEEGDataset_multisub(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', seed=42):

        file_list = self.scandir(data_root, split, str(seed))
        sub_count = len(file_list)
        self.data_list = []
        self.label_list = []
        self.lengths = []
        self.data_len = 0 
        self.channels = []
        for file in file_list:
            print(file)
            data = torch.load(file)['samples']
            label = torch.load(file)['labels']
            self.data_list.append(data)
            self.label_list.append(label)
            self.lengths.append(data.shape[0])
            self.channels.append(data.shape[1])
            if(data.shape[0]> self.data_len):
                self.data_len = data.shape[0]
    def __getitem__(self, index):
        # start_time = time.time()
        data = []
        labels = []
        for seeg, label in zip(self.data_list, self.label_list):
            index_mod = index % seeg.shape[0]
            data.append(seeg[index_mod])
            labels.append(label[index_mod])
        # end_time = time.time()
        # data_preparation_time = end_time - start_time
        
        # print(f"data time: {data_preparation_time:.3f} seconds")
        return data, labels
    
    def scandir(self, directory, split, seed):
        matching_files = []

        # Walk through all files in the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if both keywords are in the file name
                if split in file:
                    file_path = os.path.join(root, file)
                    matching_files.append(file_path)
        return matching_files
    def __len__(self):
        return self.data_len

class sEEGDataset_multisub_masking(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', seed=42):

        file_list = self.scandir(data_root, split, str(seed))
        self.sub_count = len(file_list)
        self.data_list = []
        self.label_list = []
        self.mask_list = []
        self.lengths = []
        self.data_len = 0 
        self.channels = []
        self.region_indeces = []
        self.region_counts = []
        for file in file_list:
            print(file)
            dataset = torch.load(file)
            data = dataset['samples']
            label = dataset['labels']
            # mask = dataset['masks']
            region_index = dataset['brain_region_indeces']
            brain_region_count = dataset['brain_region_count']
            self.region_counts.append(brain_region_count)
            self.region_indeces.append(region_index)
            self.data_list.append(data)
            self.label_list.append(label)
            # self.mask_list.append(mask)
            self.lengths.append(data.shape[0])
            self.channels.append(data.shape[1])

            if(data.shape[0]> self.data_len):
                self.data_len = data.shape[0]
        label_all_sub = flat_list = [item[0].item() for sublist in self.label_list for item in sublist]
        self.class_weights = compute_class_weight('balanced', classes=np.unique(label_all_sub), y=label_all_sub)
    def __getitem__(self, index):
        # start_time = time.time()
        data = []
        labels = []
        masks = []
        # masks = []
        # pivot = [random.randint(0, self.data_len) for _ in range(self.sub_count)]
        # for i, (seeg, label, mask) in enumerate(zip(self.data_list, self.label_list, self.mask_list)):
        #     # index_mod = (index + pivot[i]) % seeg.shape[0]
        #     index_mod = index % seeg.shape[0]
        #     data.append(seeg[index_mod])
        #     labels.append(label[index_mod])
        #     # masks.append(mask[index_mod])
        
        # return data, labels, masks

        for i, (seeg, label) in enumerate(zip(self.data_list, self.label_list)):
            # index_mod = (index + pivot[i]) % seeg.shape[0]
            index_mod = index % seeg.shape[0]
            data.append(seeg[index_mod])
            labels.append(label[index_mod])
            import random
            # Parameters
            num_range = 21  # Values from 0 to 20
            probability = 0.1  # 10% chance
            # Generate the list based on the given probability
            mask = [1 if random.random() < probability else 0 for _ in range(num_range)]
            masks.append(mask)
        
        return data, labels, masks
    
    
    def scandir(self, directory, split, seed):
        matching_files = []

        # Walk through all files in the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if both keywords are in the file name
                if(split == 'all'):
                    if split in file:
                        file_path = os.path.join(root, file)
                        matching_files.append(file_path)
                else:
                    if split in file and seed in file:
                        file_path = os.path.join(root, file)
                        matching_files.append(file_path)
        return matching_files
    def __len__(self):
        return self.data_len

class PaddedSEEGDataset_multisub(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', seed=42):

        file_list = self.scandir(data_root, split, str(seed))
        self.sub_count = len(file_list)
        self.data_list = []
        self.label_list = []
        self.mask_list = []
        self.lengths = []
        self.data_len = 0 
        self.channels = []
        self.region_indeces = []
        self.region_counts = []
        for file in file_list:
            print(file)
            dataset = torch.load(file)
            data = dataset['samples']
            label = dataset['labels']
            # mask = dataset['masks']
            region_index = dataset['brain_region_indeces']
            brain_region_count = dataset['brain_region_count']
            self.region_counts.append(brain_region_count)
            self.region_indeces.append(region_index)
            self.data_list.append(data)
            self.label_list.append(label)
            # self.mask_list.append(mask)
            self.lengths.append(data.shape[0])
            self.channels.append(data.shape[1])

            if(data.shape[0]> self.data_len):
                self.data_len = data.shape[0]
        label_all_sub = flat_list = [item[0].item() for sublist in self.label_list for item in sublist]
        self.class_weights = compute_class_weight('balanced', classes=np.unique(label_all_sub), y=label_all_sub)
    def __getitem__(self, index):
        # start_time = time.time()
        data = []
        labels = []

        for i, (seeg, label) in enumerate(zip(self.data_list, self.label_list)):
            # index_mod = (index + pivot[i]) % seeg.shape[0]
            index_mod = index % seeg.shape[0]
            data.append(seeg[index_mod])
            labels.append(label[index_mod])
        
        return data, labels
    
    
    def scandir(self, directory, split, seed):
        matching_files = []

        # Walk through all files in the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if both keywords are in the file name
                if(split == 'all'):
                    if split in file:
                        file_path = os.path.join(root, file)
                        matching_files.append(file_path)
                else:
                    if split in file and seed in file:
                        file_path = os.path.join(root, file)
                        matching_files.append(file_path)
        return matching_files
    def __len__(self):
        return self.data_len


def collate_fn(batch):
    sub_count = len(batch[0][0])
    seeg_list = [[] for i in range(sub_count)]
    label_list = [[] for i in range(sub_count)]
    
    print(type(batch))
    for sample, label in batch:
        for i in range(sub_count):
            seeg_list[i].append(sample[i])
            label_list[i].append(label[i])
    
    all_tuple = ((torch.stack(seeg_list[0]), torch.LongTensor(label_list[0])), )
    for i in range(1, sub_count):
        all_tuple = all_tuple + (((torch.stack(seeg_list[i]), torch.LongTensor(label_list[i]))),)
    return all_tuple
              
def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    # print(out)
    tensor_min = out.min()
    tensor_max = out.max()
    normalized_out = (out - tensor_min) / (tensor_max - tensor_min)

    Q = torch.exp(normalized_out / epsilon).permute(0, 2, 1) # Q is K-by-B for consistency with notations from our paper
    print(Q)
    B = Q.shape[2] # number of samples to assign
    K = Q.shape[1] # how many prototypes

    # make the matrix sums to 1
    sum_Q = Q.sum(dim=(1, 2), keepdim=True)

    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    print(Q)
    # print(torch.argmax(Q.permute(0, 2, 1), dim=2, keepdim=True))
    return torch.argmax(Q.permute(0, 2, 1), dim=2, keepdim=True)

def calculate_module_size_in_gb(module):
    """Calculate the size of a PyTorch module in GB."""
    total_params = 0
    for param in module.parameters():
        total_params += param.numel() * param.element_size()
    return total_params / (1024 ** 3)  # Convert bytes to GB

# Function to rearrange X and Y into sub-arrays by subject labels
def group_by_subject(X, Y):
    # Extract subject labels
    subject_labels = Y[:, -1]

    # Get unique subject labels
    unique_labels = torch.unique(subject_labels)

    # Create sub-tensors for each subject
    X_groups = []
    Y_groups = []

    for label in unique_labels:
        indices = (subject_labels == label).nonzero(as_tuple=True)[0]
        X_groups.append(X[indices])
        Y_groups.append(Y[indices])

    return X_groups, Y_groups