import os
import zipfile

import gdown
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image


def one_hot(labels, class_size):
    """
    Creates a one-hot encoded tensor from a tensor of labels.
    
    This function transforms a tensor of integer labels into a one-hot encoded representation.
    Each label is converted into a vector where the element corresponding to the label's class
    is set to 1, and all other elements are set to 0. This is commonly used to represent
    categorical data in a format suitable for machine learning models. The resulting tensor
    is then placed on the available device (GPU if available, otherwise CPU).
    
    Args:
        labels (torch.Tensor): A tensor of integer labels.
        class_size (int): The number of classes.
    
    Returns:
        torch.Tensor: A one-hot encoded tensor on the specified device.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


def create_dataset(
    path_to_data, target_shape, batch_size=32, crop_size=128, num_of_channels=1
):
    """
    Creates a dataset and dataloader containing images of a specific shape.
    
    This method prepares image data for use in a machine learning model by applying transformations like resizing, cropping, and converting to tensors. It then filters the dataset to include only images matching the specified target shape, and creates a PyTorch DataLoader for efficient batching and loading of the data during training or evaluation.
    
    Args:
        path_to_data (str): The path to the directory containing the image data.
        target_shape (int): The integer label representing the shape to filter the images by.
        batch_size (int, optional): The batch size for the dataloader. Defaults to 32.
        crop_size (int, optional): The size to crop the images to. Defaults to 128.
        num_of_channels (int, optional): The number of color channels. Defaults to 1.
    
    Returns:
        tuple[TensorDataset, DataLoader]: A tuple containing the filtered TensorDataset and the corresponding DataLoader.
    """
    transform = transforms.Compose(
        [
            # transforms.RandomRotation(degrees=(0, 360)),
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
    '''
    A Variational Autoencoder (VAE) class for conditional image generation.
    
        Attributes:
            encoder: The encoder network.
            decoder: The decoder network.
            latent_size: The dimensionality of the latent space.
            class_size: The number of classes.
    
        Methods:
        - encode: Encodes the input x and class c into the latent space, returning the mean and log variance.
        - reparameterize: Reparameterizes the latent space using the mean and log variance.
        - decode: Decodes a latent vector z and class c back into the image space.
        - forward: Performs the complete encoding and decoding process.
    '''

    def __init__(self, feature_size, latent_size, class_size):
        """
        Initializes the Conditional Variational Autoencoder (CVAE).
        
        This constructor defines the neural network architecture for encoding input data into a latent space and decoding it back into its original form, conditioned on class information. This allows the model to learn a compressed, meaningful representation of the data while being able to generate new samples from specific classes.
        
        Args:
            feature_size (int): The dimension of the input feature vector.
            latent_size (int): The dimension of the latent space.
            class_size (int): The number of classes for conditional generation.
        
        Returns:
            None
        
        Initializes the following class fields:
            feature_size: Stores the size of the input feature vector.
            class_size: Stores the number of classes.
            fc1: The first fully connected layer in the encoder.
            fc21: The first linear layer for the mean in the encoder.
            fc22: The second linear layer for the standard deviation in the encoder.
            fc3: The first fully connected layer in the decoder.
            fc4: The output layer of the decoder.
            elu: The ELU activation function.
            sigmoid: The Sigmoid activation function.
        """
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1 = nn.Linear(feature_size + class_size, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 512)
        self.fc4 = nn.Linear(512, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):  # Q(z|x, c)
        """
        x: (bs, feature_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes a distribution using the mean and log variance.
        
        This function generates a sample from a Gaussian distribution with provided mean and variance.
        The reparameterization trick is employed to enable gradient-based optimization.
        
        Args:
            mu (torch.Tensor): The mean of the Gaussian distribution.
            logvar (torch.Tensor): The log variance of the Gaussian distribution.
        
        Returns:
            torch.Tensor: The reparameterized sample drawn from the Gaussian distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        """
        z: (bs, latent_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        """
        Encodes the input, samples from the latent space, and decodes to reconstruct the data.
        
        Args:
            x: The input tensor representing the data to be encoded and decoded.
            c: The conditioning variable providing additional information to guide the encoding and decoding process.
        
        Returns:
            A tuple containing:
                The reconstructed output from the decoder, the mean (mu) of the latent distribution, and the log variance (logvar) of the latent distribution.
        """
        crop_size = 128
        mu, logvar = self.encode(x.view(-1, crop_size * crop_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def inference(target_shape="Cube"):
    """
    Generates samples using a pre-trained Conditional Variational Autoencoder (CVAE).
    
    This method retrieves a dataset and a CVAE checkpoint, loads the model, and then 
    generates images corresponding to the specified target shape. The generated images
    are saved to a designated results directory. This enables exploring the latent space 
    of the trained model and visualizing the generated outputs.
    
    Args:
        target_shape (str, optional): The desired shape for the generated samples. 
            Valid options are "cube", "sphere", "stick", "flat", and "amorphous". 
            Defaults to "Cube".
    
    Returns:
        None: The method saves the generated images to disk and does not return a value.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kwargs = {"num_workers": 1, "pin_memory": True}

    # hyper params
    latent_size = 512
    crop_size = 128

    path_to_data = os.environ.get("PATH_TO_DATA")
    if not os.path.exists(path_to_data):
        print(f"Dataset not found. Downloading...")
        os.makedirs(os.path.dirname(path_to_data), exist_ok=True)

        zip_file_path = os.path.join(
            os.path.dirname(path_to_data), "image_dataset_multi_filtered.zip"
        )
        gdrive_file_id = "1TqZhMh9lYE0ziJnf_g7Rfs2OoC095hJ1"  # Extracted from URL

        gdown.download(id=gdrive_file_id, output=zip_file_path, quiet=False)

        # Extract ZIP contents
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(path_to_data))

        # Rename extracted folder if necessary (ensure it matches path_to_data)
        extracted_folder = os.path.join(
            os.path.dirname(path_to_data), zip_ref.namelist()[0].split("/")[0]
        )
        if extracted_folder != path_to_data:
            os.rename(extracted_folder, path_to_data)

        # Remove the ZIP file
        os.remove(zip_file_path)

        print(f"Dataset extracted to: {path_to_data}")

    model = CVAE(crop_size * crop_size, latent_size, 5).to(device)

    target_shape = target_shape.lower()
    classes = {"cube": 0, "sphere": 1, "stick": 2, "flat": 3, "amorphous": 4}
    for shape in classes.keys():
        if shape in target_shape:
            target_shape = shape
            break

    dataset, train_loader = create_dataset(
        path_to_data,
        classes[target_shape],
        batch_size=5,
        crop_size=crop_size,
        num_of_channels=1,
    )
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

    weights_only = torch.cuda.is_available()

    path_to_checkpoint = os.environ.get("PATH_TO_CVAE_CHECKPOINT")
    if not os.path.isfile(path_to_checkpoint):
        # raise FileNotFoundError(f"File '{path_to_checkpoint}' does not exist.")
        print(f"File '{path_to_checkpoint}' does not exist. Downloading checkpoint...")

        directory = os.path.dirname(path_to_checkpoint)
        os.makedirs(directory, exist_ok=True)
        gdown.download(
            url="https://drive.google.com/uc?id=1F5hj9HRsauvR2DwYduL2e-DzHdOPk9j1",
            output=path_to_checkpoint,
        )

    model.load_state_dict(
        torch.load(path_to_checkpoint, weights_only=weights_only, map_location=device)[
            "model_state_dict"
        ]
    )
    model.eval()

    path_to_results = os.path.join(os.environ.get("PATH_TO_RESULTS"), "cvae")
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)

    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, 5)
            mu, logvar = model.encode(data.view(-1, crop_size * crop_size), labels)
            z = model.reparameterize(mu, logvar)
            sample = torch.randn(5, latent_size).to(device)
            sample = 0.75 * z + sample * 0.25
            sample = model.decode(sample, labels).cpu()
            save_image(
                sample.view(5, 1, crop_size, crop_size),
                os.path.join(path_to_results, "sample_" + str(f"{i:02}") + ".png"),
            )


if __name__ == "__main__":
    from dotenv import load_dotenv

    from definitions import CONFIG_PATH

    load_dotenv(CONFIG_PATH)

    inference()
    print()
