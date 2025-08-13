import torch
import torch.utils.data
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset

import os
import gdown
import zipfile


def one_hot(labels, class_size):
    """
    Creates a one-hot encoded tensor from a tensor of labels.
    
    Args:
        labels (torch.Tensor): A tensor of class labels (integers).
        class_size (int): The number of classes.
    
    Returns:
        torch.Tensor: A one-hot encoded tensor on the appropriate device (CUDA if available, otherwise CPU).
    
    This method transforms a tensor of class labels into a one-hot encoded representation. 
    Each label in the input tensor is converted into a vector where the element corresponding to the label's index is set to 1, 
    while all other elements are 0. This representation is commonly used in machine learning tasks, 
    particularly in classification problems, to represent categorical data in a numerical format suitable for training models.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

def create_dataset(path_to_data, target_shape, batch_size=32, crop_size=128, num_of_channels=1):
    """
    Creates a dataset and dataloader for a given path and target shape.
    
    This method prepares a PyTorch dataset from images, applies transformations 
    like resizing, cropping, grayscale conversion, and filters images based on a 
    specified label to create a focused dataset suitable for model training or 
    evaluation. It then generates a dataloader for efficient data handling.
    
    Args:
        path_to_data (str): The path to the directory containing the image data.
        target_shape (int): The target shape (label) to filter the images by.
        batch_size (int, optional): The desired batch size for the dataloader. Defaults to 32.
        crop_size (int, optional): The size to which images will be cropped. Defaults to 128.
        num_of_channels (int, optional): The number of output channels for grayscale conversion. Defaults to 1.
    
    Returns:
        tuple[TensorDataset, DataLoader]: A tuple containing the TensorDataset and DataLoader objects.
    """
    transform = transforms.Compose(
        [
            #transforms.RandomRotation(degrees=(0, 360)),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=num_of_channels),
        ]
    )
    dataset = ImageFolder(root=path_to_data, transform=transform)
    images = torch.Tensor([])
    labels = torch.IntTensor([])

    for image, label in dataset:
        if label == target_shape:
            images = torch.cat((images, image))
            labels = torch.cat((labels, torch.IntTensor([label])))
    dataset = TensorDataset(images, labels)
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=g,
    )
    return dataset, dataloader

class CVAE(nn.Module):
    """
    CVAE implementation.
    
        This class implements a Conditional Variational Autoencoder (CVAE) for generating data 
        conditioned on class labels. It consists of an encoder, a reparameterization trick, 
        and a decoder.
    
        Attributes:
            encoder: The encoder network.
            decoder: The decoder network.
            latent_size: The dimensionality of the latent space.
            class_size: The number of classes.
    
        Methods:
        - __init__: Initializes the CVAE with encoder and decoder networks, and latent/class sizes.
        - encode: Encodes the input data and class label into the latent space, returning mean and log variance.
        - reparameterize: Applies the reparameterization trick to sample from the latent distribution.
        - decode: Decodes a latent vector and class label into the output space.
        - forward: Performs a complete forward pass through the CVAE, encoding, reparameterizing, and decoding.
    """

    def __init__(self, feature_size, latent_size, class_size):
        """
        Initializes the Conditional Variational Autoencoder (CVAE). This model learns a compressed, latent representation of input features conditioned on class labels, enabling the generation of new data points similar to the training data.
        
        Args:
            feature_size (int): The size of the input feature vector.
            latent_size (int): The dimension of the latent space.
            class_size (int): The number of classes.
        
        Fields initialized:
            feature_size: Stores the size of the input feature vector.
            class_size: Stores the number of classes.
            fc1: First fully connected layer for encoding, taking both features and class information as input.
            fc21: First linear layer for the mean of the latent distribution.
            fc22: Second linear layer for the standard deviation of the latent distribution.
            fc3: First fully connected layer for decoding, combining the latent vector and class information.
            fc4: Second linear layer for reconstructing the feature vector.
            elu: ELU activation function applied during encoding and decoding.
            sigmoid: Sigmoid activation function used to constrain the output values.
        
        Returns:
            None
        """
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 512)
        self.fc4 = nn.Linear(512, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes a distribution by sampling from a Gaussian.
        
        This function enables gradient-based learning in models that involve stochastic sampling. By expressing a random sample as a deterministic function of the mean, variance, and a random noise variable, it allows gradients to flow through the sampling process.
        
        Args:
            mu (torch.Tensor): The mean of the Gaussian distribution.
            logvar (torch.Tensor): The log variance of the Gaussian distribution.
        
        Returns:
            torch.Tensor: A sample from the Gaussian distribution.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        """
        Encodes the input, samples from the latent space, and decodes to reconstruct the data.
        
        Args:
            x: The input tensor representing the data to be encoded and decoded.
            c: The conditioning variable providing additional context for encoding and decoding.
        
        Returns:
            A tuple containing:
                The decoded output tensor, the mean (mu) from the encoder, and the log variance (logvar) from the encoder.
        """
        crop_size = 128
        mu, logvar = self.encode(x.view(-1, crop_size*crop_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def inference(target_shape = 'Cube'):
    """
    Generates new data samples using a pre-trained Conditional Variational Autoencoder (CVAE).
    
    This method facilitates the creation of novel data points by utilizing a pre-trained CVAE model. 
    It handles data loading, model initialization, and sample generation based on a specified target shape, 
    ultimately saving the results as image files.
    
    Args:
        target_shape (str, optional): The desired shape for the generated samples. Defaults to 'Cube'.
            Supported shapes are 'cube', 'sphere', 'stick', 'flat', and 'amorphous'.
    
    Returns:
        None
    
    Class Fields Initialized:
        - device: A PyTorch device object ('cuda' if available, otherwise 'cpu').
        - kwargs: A dictionary containing keyword arguments for the DataLoader (num_workers=1, pin_memory=True).
        - latent_size: Integer representing the dimensionality of the latent space (512).
        - crop_size: Integer representing the size of the input images (128).
        - path_to_data: Path to the dataset directory.
        - model: An instance of the CVAE model.
        - classes: A dictionary mapping shape names to their corresponding integer labels.
        - dataset: The loaded dataset.
        - train_loader: The DataLoader for the dataset.
        - weights_only: Boolean indicating whether to load only the model weights.
        - path_to_checkpoint: Path to the CVAE checkpoint file.
        - path_to_results: Path to the directory where generated samples are saved.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True}

    # hyper params
    latent_size = 512
    crop_size = 128

    path_to_data = os.environ.get('PATH_TO_DATA')
    if not os.path.exists(path_to_data):
        print(f"Dataset not found. Downloading...")
        os.makedirs(os.path.dirname(path_to_data), exist_ok=True)

        zip_file_path = os.path.join(os.path.dirname(path_to_data), "image_dataset_multi_filtered.zip")
        gdrive_file_id = "1TqZhMh9lYE0ziJnf_g7Rfs2OoC095hJ1"  # Extracted from URL

        gdown.download(id=gdrive_file_id, output=zip_file_path, quiet=False)

        # Extract ZIP contents
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(path_to_data))

        # Rename extracted folder if necessary (ensure it matches path_to_data)
        extracted_folder = os.path.join(os.path.dirname(path_to_data), zip_ref.namelist()[0].split('/')[0])
        if extracted_folder != path_to_data:
            os.rename(extracted_folder, path_to_data)

        # Remove the ZIP file
        os.remove(zip_file_path)

        print(f"Dataset extracted to: {path_to_data}")

    model = CVAE(crop_size*crop_size, latent_size, 5).to(device)

    target_shape = target_shape.lower()
    classes = {'cube': 0, 'sphere': 1, 'stick': 2, 'flat': 3, 'amorphous': 4}
    for shape in classes.keys():
        if shape in target_shape:
            target_shape = shape
            break 
        
    dataset, train_loader = create_dataset(path_to_data, classes[target_shape], batch_size=5, crop_size=crop_size, num_of_channels=1)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

    weights_only = torch.cuda.is_available()

    path_to_checkpoint = os.environ.get('PATH_TO_CVAE_CHECKPOINT')
    if not os.path.isfile(path_to_checkpoint):
        #raise FileNotFoundError(f"File '{path_to_checkpoint}' does not exist.")
        print(f"File '{path_to_checkpoint}' does not exist. Downloading checkpoint...")
        
        directory = os.path.dirname(path_to_checkpoint)
        os.makedirs(directory, exist_ok=True)
        gdown.download(url='https://drive.google.com/uc?id=1F5hj9HRsauvR2DwYduL2e-DzHdOPk9j1', output=path_to_checkpoint)
    
    model.load_state_dict(torch.load(path_to_checkpoint, weights_only=weights_only, map_location=device)['model_state_dict'])
    model.eval()

    path_to_results = os.path.join(os.environ.get('PATH_TO_RESULTS'), 'cvae')
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)

    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, 5)
            mu, logvar = model.encode(data.view(-1, crop_size*crop_size), labels)
            z = model.reparameterize(mu, logvar)
            sample = torch.randn(5, latent_size).to(device)
            sample = 0.75*z+sample*0.25
            sample = model.decode(sample, labels).cpu()
            save_image(sample.view(5, 1, crop_size, crop_size), os.path.join(path_to_results, 'sample_' + str(f"{i:02}") + '.png'))
