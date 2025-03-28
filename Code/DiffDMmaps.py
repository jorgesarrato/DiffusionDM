import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from score_models import ScoreModel, NCSNpp
from tqdm import tqdm
import numpy.ma as ma

def minmaxnorm(arr, accept_nans = False):
    if accept_nans:
        return (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))
    else:
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def stdnorm(arr, accept_nans = False):
    if accept_nans:
        return (arr-np.nanmean(arr))/np.nanstd(arr)
    else:
        return (arr-np.mean(arr))/np.std(arr)


class CustomDataset(Dataset):
    def __init__(self, folder_path, name_roots, N, N_aug, allow_nans, device='cpu', binned = True):
        """
        Args:
            folder_path (str): Path to the folder containing the .npy files.
            name_roots (list): List of name roots for the files (e.g., ['file_Xvel', 'file_Xcount', 'file_XDMmass', 'file_Xstd']).
            N (int): Number of files to read for each type.
            N_aug (int): Number of augmentations to select from each file.
            device (str): Device to send the tensors to (e.g., 'cpu' or 'cuda').
        """
        self.folder_path = folder_path
        self.name_roots = name_roots
        self.N = N
        self.N_aug = N_aug
        self.device = device
        self.allow_nans = allow_nans
        self.binned = binned
        self.data = self._load_data()

    def _load_data(self):
        data = {name_root: [] for name_root in self.name_roots}
        
        
        for i in tqdm(range(1, self.N + 1)):
            selected_indices = set()
            for name_root in self.name_roots:
                selected_augmentations = []
                file_path = os.path.join(self.folder_path, f"{name_root}{i}.npy")
                if os.path.exists(file_path):
                    array = np.load(file_path)  # Shape: (32, 48, 48)
                    if self.allow_nans:
                        valid_indices = list(range(array.shape[0]))
                    else:
                        valid_indices = list(np.where(np.all(~np.isnan(array), axis = (1,2)))[0])
                    if name_root == self.name_roots[0]:
                        indices = np.random.choice(list(valid_indices), np.min((self.N_aug, len(valid_indices))), replace=False)
                        for idx in indices:
                            selected_augmentations.append(array[idx])
                            selected_indices.add(idx)
                        if len(selected_augmentations) > 0:
                            data[name_root].append(np.array(selected_augmentations))
                    else:
                        selected_augmentations = [array[idx] for idx in selected_indices]
                        if len(selected_augmentations) > 0:
                            data[name_root].append(np.array(selected_augmentations))
                else:
                    print(f"File {file_path} not found.")
        
        # Stack all selected augmentations for each name_root
        for name_root in self.name_roots:
            if data[name_root]:
                data[name_root] = np.vstack(data[name_root])  # Shape: (N * N_aug, 48, 48)
            else:
                data[name_root] = np.array([])

        self._postprocess_all_maps(data)
        
        return data
    

    def _postprocess_all_maps(self, data):
        """
        Postprocesses all maps in the dataset according to the specified rules.

        Args:
            data (dict): Dictionary containing raw maps for each name root.
        """
        # Get all maps
        if self.binned:
            dm_maps = data['file_XDMmass_binned']  # Shape: (N * N_aug, 48, 48)
            vel_maps = data['file_Xvel_binned']    # Shape: (N * N_aug, 48, 48)
            count_maps = data['file_Xcount_binned']  # Shape: (N * N_aug, 48, 48)
            std_maps = data['file_Xstd_binned']    # Shape: (N * N_aug, 48, 48)
        else:
            dm_maps = data['file_XDMmass']  # Shape: (N * N_aug, 48, 48)
            vel_maps = data['file_Xvel']    # Shape: (N * N_aug, 48, 48)
            count_maps = data['file_Xcount']  # Shape: (N * N_aug, 48, 48)
            std_maps = data['file_Xstd']    # Shape: (N * N_aug, 48, 48)
        
        # Postprocess each map
        for i in range(len(dm_maps)):
            # Count map: Divide by the maximum value in the map
            count_maps[i] = minmaxnorm(count_maps[i], accept_nans=self.allow_nans)
            
            # DM map: Take the base 10 logarithm and divide by the maximum value in the map
            dm_maps[i] = np.log10(dm_maps[i])
            dm_maps[i] = minmaxnorm(dm_maps[i], accept_nans=self.allow_nans)

            # Vel map: Divide by the maximum absolute value across all vel maps and replace NaNs with -2
            #max_abs_vel = np.nanmax(np.abs(vel_maps))
            #vel_maps[i] = vel_maps[i] / max_abs_vel
            vel_maps[i] = minmaxnorm(vel_maps[i], accept_nans=self.allow_nans)

            # Std map: Divide by the maximum absolute value across all std maps and replace NaNs with -2
            #max_abs_std = np.nanmax(np.abs(std_maps))
            #std_maps[i] = std_maps[i] / max_abs_std
            std_maps[i] = minmaxnorm(std_maps[i], accept_nans=self.allow_nans)

            if allow_nans:
                count_maps[i] = np.nan_to_num(count_maps[i], nan=-1)
                dm_maps[i] = np.nan_to_num(dm_maps[i], nan=-1)
                vel_maps[i] = np.nan_to_num(vel_maps[i], nan=-1)
                std_maps[i] = np.nan_to_num(std_maps[i], nan=-1)

    def __len__(self):
        return len(self.data[self.name_roots[0]])

    def __getitem__(self, idx):
        # Get DMmap as 1x48x48 tensor
        mass_str = 'file_XDMmass_binned' if self.binned else 'file_XDMmass'
        dm_map = torch.tensor(self.data[mass_str][idx], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 48, 48)
        
        # Get the other arrays except for 'file_XDMmass' as a 3x48x48 tensor
        other_arrays = np.stack([
            self.data[name_root][idx] for name_root in self.name_roots if name_root != mass_str
        ], axis=0)  # Shape: (3, 48, 48)
        other_arrays = torch.tensor(other_arrays, dtype=torch.float32)
        
        # Send tensors to the specified device
        dm_map = dm_map.to(self.device)
        other_arrays = other_arrays.to(self.device)
        
        return dm_map, other_arrays

def plot_dataset_element(dataset, idx, cmap='viridis', minmax = True ):
    """
    Plots the arrays for a given element of the dataset.

    Args:
        dataset (CustomDataset): The dataset object.
        idx (int): Index of the element to plot.
        cmap (str): Colormap to use for the images.
    """

    if minmax:
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_under('white')  # Values under vmin will be white

    # Get the element from the dataset
    dm_map, other_arrays = dataset[idx]
    
    # Convert tensors to numpy arrays
    dm_map_np = dm_map.cpu().numpy().squeeze()  # Shape: (48, 48)
    other_arrays_np = other_arrays.cpu().numpy()  # Shape: (N, 48, 48)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, other_arrays_np.shape[0] + 1, figsize=(5 * (other_arrays_np.shape[0] + 1), 5))
    
    # Plot DMmap
    ax = axes[0]
    masked_dm_map = np.ma.masked_where(np.isnan(dm_map_np), dm_map_np)
    if minmax:
        im = ax.imshow(masked_dm_map, cmap=cmap, interpolation='none', vmin = 0, vmax = 1)
    else:
        im = ax.imshow(masked_dm_map, cmap=cmap, interpolation='none')
    if np.isnan(dm_map_np).any():
        nan_mask = np.isnan(dm_map_np)
        ax.imshow(np.where(nan_mask, 1, np.nan), cmap='Reds', alpha=0.5, interpolation='none')
    ax.set_title('DMmap')
    fig.colorbar(im, ax=ax, orientation='vertical')
    
    # Plot the other arrays
    for i in range(other_arrays_np.shape[0]):
        ax = axes[i + 1]
        array_np = other_arrays_np[i]
        masked_array = np.ma.masked_where(np.isnan(array_np), array_np)
        if minmax:
            im = ax.imshow(masked_array, cmap=cmap, interpolation='none', vmin = 0 ,vmax = 1)
        else:
            im = ax.imshow(masked_array, cmap=cmap, interpolation='none')
        if np.isnan(array_np).any():
            nan_mask = np.isnan(array_np)
            ax.imshow(np.where(nan_mask, 1, np.nan), cmap='Reds', alpha=0.5, interpolation='none')
        ax.set_title(f'Array {i + 1}')
        fig.colorbar(im, ax=ax, orientation='vertical')
    
    plt.tight_layout()
    plt.show()


folder_path = '/home/jsarrato/Physics/PhD/Paper-DiffusionDM/newmaps'
# Example usage:
#folder_path = '/net/debut/scratch/jsarrato/Wolf_for_FIRE/work/Maps_Dispersion_NIHAO_noPDF_DMmap_SPH'
#name_roots = ['file_XDMmass', 'file_Xvel', 'file_Xcount', 'file_Xstd','file_XDMmass_binned', 'file_Xvel_binned', 'file_Xcount_binned', 'file_Xstd_binned', 'file_Xstd_binned_alt']
name_roots = ['file_XDMmass_binned', 'file_Xvel_binned', 'file_Xcount_binned', 'file_Xstd_binned']

N = 1000  # Number of files to read for each type
N_aug = 16  # Number of augmentations to select from each file
allow_nans = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

binned = True
dataset = CustomDataset(folder_path, name_roots, N, N_aug, allow_nans, device = device, binned = binned)

if binned:
    print(np.any(np.isnan(dataset.data['file_XDMmass_binned'])))
    print(np.any(np.isnan(dataset.data['file_Xvel_binned'])))
    print(np.any(np.isnan(dataset.data['file_Xcount_binned'])))
    print(np.any(np.isnan(dataset.data['file_Xstd_binned'])))
else:
    print(np.any(np.isnan(dataset.data['file_XDMmass'])))
    print(np.any(np.isnan(dataset.data['file_Xvel'])))
    print(np.any(np.isnan(dataset.data['file_Xcount'])))
    print(np.any(np.isnan(dataset.data['file_Xstd'])))
# Example usage:
# Assuming `dataset` is already created using the CustomDataset class
plot_dataset_element(dataset, idx=0)  # Plot the first element in the dataset


net = NCSNpp(channels=1, dimensions=2, condition=['input'], condition_input_channels=3).to(device)



config = {"BETA_MIN": 1e-2,
    "BETA_MAX": 20.,
    "SIGMA_MIN": 1e-2,
    "SIGMA_MAX": 50.}

sigma = False

if sigma:
        model = ScoreModel(model=net, sigma_min=config['SIGMA_MIN'], sigma_max=config['SIGMA_MAX']).to(device)
else:
        model = ScoreModel(model=net, beta_min=config['BETA_MIN'], beta_max=config['BETA_MAX']).to(device)


# Define any preprocessing function if needed
def preprocessing_fn(x):
    return x

# Set the hyperparameters and other options for training
learning_rate = 1e-6
batch_size = 32
epochs = 100
checkpoints_directory = f"models_N{N}_N_aug{N_aug}_sigma{sigma}"
checkpoints = 10 # save a checkpoint every 10 epochs
models_to_keep = 1 # only keep one model, erase previous ones
seed = 42

# Fit the model to the dataset
losses = model.fit(
    dataset,
    preprocessing_fn=preprocessing_fn,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs,
    checkpoints_directory=checkpoints_directory,
    checkpoints=checkpoints,
    models_to_keep=models_to_keep,
    seed=seed
    )



# Save the loss plot to the checkpoints directory
plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
loss_plot_path = os.path.join(checkpoints_directory, 'loss_plot.png')
plt.savefig(loss_plot_path)
plt.close()


# Number of random indices to select
num_indices = 10

# Randomly select 10 indices from the dataset
random_indices = np.random.choice(len(dataset), num_indices, replace=False)

for i, index_n in enumerate(random_indices):
    # Generate samples for the current index
    samples_el_1 = model.sample(
        shape=(100, 1, 48, 48),
        steps=1000,
        condition=[torch.tensor(np.repeat(dataset.__getitem__(index_n)[1].reshape(1, 3, 48, 48).cpu().numpy(), 100, axis=0)).to(device)]
    ).cpu().numpy()

    # Calculate median and standard deviation of the samples
    median_prediction_1 = np.median(samples_el_1, axis=0)[0, :, :]
    std_prediction_1 = np.std(samples_el_1, axis=0)[0, :, :]

    # Save the samples to a file
    samples_path = os.path.join(checkpoints_directory, f'samples_{i}.npy')
    np.save(samples_path, samples_el_1)

    # Get the label map from the current element of the dataset
    label_map = dataset.__getitem__(index_n)[0].cpu().numpy().squeeze()

    # Plot the label map, median prediction, and standard deviation prediction
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Function to calculate the lowest positive value in an image
    def get_lowest_positive_value(image):
        positive_values = image[image > 0]
        return np.min(positive_values) if positive_values.size > 0 else 1.0  # Default to 1.0 if no positive values

    # Plot the label map
    ax = axes[0]
    vmin_label = get_lowest_positive_value(label_map)
    masked_label_map = ma.masked_where(label_map <= 0, label_map)  # Mask non-positive values
    im = ax.imshow(masked_label_map, cmap='viridis', interpolation='none', vmin=vmin_label)
    ax.set_title(f'Label Map (Index {index_n})')
    fig.colorbar(im, ax=ax, orientation='vertical')

    # Plot the median prediction
    ax = axes[1]
    vmin_median = get_lowest_positive_value(median_prediction_1)
    masked_median = ma.masked_where(median_prediction_1 <= 0, median_prediction_1)  # Mask non-positive values
    im = ax.imshow(masked_median, cmap='viridis', interpolation='none', vmin=vmin_median)
    ax.set_title('Median Prediction')
    fig.colorbar(im, ax=ax, orientation='vertical')

    # Plot the standard deviation prediction
    ax = axes[2]
    vmin_std = get_lowest_positive_value(std_prediction_1)
    masked_std = ma.masked_where(std_prediction_1 <= 0, std_prediction_1)  # Mask non-positive values
    im = ax.imshow(masked_std, cmap='viridis', interpolation='none', vmin=vmin_std)
    ax.set_title('Standard Deviation Prediction')
    fig.colorbar(im, ax=ax, orientation='vertical')

    # Save the comparison plot to the checkpoints directory
    comparison_plot_path = os.path.join(checkpoints_directory, f'comparison_plot_{i}.png')
    plt.savefig(comparison_plot_path)
    plt.close()

print(f"Training complete. Checkpoints and plots saved to: {checkpoints_directory}")


"""

class ExsituCondDatasets(Dataset):
    def _init_(self, dataset, cond_dataset, size=48, test_percentage=0.05, name='',
                 save_dir='', cond_noise_sigma=0.0, augment=False, transform=None,
                 key_list=None, *args):
        self.args = args
        self.name = name
        self.save_dir = save_dir
        self.size = size
        self.cond_noise_sigma = cond_noise_sigma
        self.augment = augment
        self.transform = transform
        self.key_list = key_list

        splits_path = None
        if self.save_dir:
            splits_path = os.path.join(self.save_dir, 'splits_{}'.format(self.name))

        if splits_path and os.path.exists(splits_path):
            train_indices, test_indices = load_splits(splits_path, self.name)
        else:
            # Keys are in form 'subhaloid_snapshot_view'
            # Get unique keys independent of view
            unique_keys = set([k.rsplit('_', 1)[0] for k in key_list])
            # Load training and test datasets
            test_len = int(len(unique_keys) * test_percentage)

            test_keys = set(random.sample(sorted(unique_keys), test_len))
            train_keys = unique_keys - test_keys

            test_indices = [i for i, k in enumerate(key_list)
                            if k.rsplit('_', 1)[0] in test_keys]
            train_indices = [i for i, k in enumerate(key_list)
                             if k.rsplit('_', 1)[0] in train_keys]

            if splits_path:
                save_splits(splits_path, self.name, train_indices, test_indices)

        # Split the datasets using the generated indices
        self.dataset = dataset[train_indices].unsqueeze(1).float()
        self.dataset_test = dataset[test_indices].unsqueeze(1).float()
        
        self.cond_dataset = cond_dataset[train_indices].float()
        self.cond_dataset_test = cond_dataset[test_indices].float()

        print('Dataset {} train: {}'.format(self.name, self.dataset.shape))
        print('Dataset {} conditions train: {}'.format(self.name, self.cond_dataset.shape))
        print('Dataset {} test: {}'.format(self.name, self.dataset_test.shape))
        print('Dataset {} conditions test: {}'.format(self.name, self.cond_dataset_test.shape))
    
    @staticmethod
    def add_noise(data):
        noise = torch.randn(data.size()) * 0.1
        return data + noise

    @staticmethod
    def add_gaussian_noise(data, std):
        noise = torch.normal(0, std, size=data.size())
        return data + noise
    
    @staticmethod
    def flip_image(image, dims=[1, 2]):
        return torch.flip(image, dims=dims)

    @staticmethod
    def rotate_image(image, angle, fill_val):
        return TF.rotate(image, angle, fill=fill_val)

    @staticmethod
    def get_background_value(image):
        background_list = []
        for channel in range(image.shape[0]):
            corner_pix_values = [image[channel, 0, 0], image[channel, -1, 0],
                                 image[channel, 0, -1], image[channel, -1, -1]]
            norm_zero_value = max(corner_pix_values, key=corner_pix_values.count)
            background_list.append(norm_zero_value)
        return background_list

    @staticmethod
    def standardize_data(data):
        mean = data.mean(dim=(1, 2), keepdim=True)  # Compute mean per channel
        std = data.std(dim=(1, 2), keepdim=True)    # Compute std per channel
        return (data - mean) / std

    def custom_augment(self, img, cond):
        hor_flip = random.choice([True, False])
        if hor_flip:
            img = self.flip_image(img, dims=[1])
            cond = self.flip_image(cond, dims=[1])

        ver_flip = random.choice([True, False])
        if ver_flip:
            img = self.flip_image(img, dims=[2])
            cond = self.flip_image(cond, dims=[2])

        rotation = random.randint(0, 360)
        if rotation > 10:
            bck = self.get_background_value(img)
            img = self.rotate_image(img, rotation, bck)
            bck = self.get_background_value(cond)
            cond = self.rotate_image(cond, rotation, bck)

        return img, cond

    def _len_(self):
        return len(self.dataset)

    def _getitem_(self, idx):
        img = self.dataset[idx]
        cond = self.cond_dataset[idx]

        if self.augment:
            self.custom_augment(img, cond)

        if self.transform:
            bck = self.get_background_value(img)
            cond_bck = self.get_background_value(cond)

            transform = Compose([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(p=0.5, border_mode=0, fill=bck, fill_mask=cond_bck),
                RandomResizedCrop(size=(self.size, self.size), scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.5),
            ])

            img = img.permute(1, 2, 0).cpu().numpy()
            cond = cond.permute(1, 2, 0).cpu().numpy()

            augmented = transform(image=img, mask=cond)
            img, cond = augmented["image"], augmented["mask"]

            # Convert back to PyTorch tensors (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1)
            cond = torch.from_numpy(cond).permute(2, 0, 1)

        if self.cond_noise_sigma > 0:
            cond = self.add_gaussian_noise(cond, self.cond_noise_sigma)
            cond = self.standardize_data(cond)  # Standardize again after adding the noise

        return img.to(device), cond.to(device)
"""
