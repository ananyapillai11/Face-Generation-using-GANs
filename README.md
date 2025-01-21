# Face Generation using GANs

This project demonstrates the use of Generative Adversarial Networks (GANs) to generate human face images. The GAN consists of two neural networks:
- **Generator**: Creates fake images from random noise.
- **Discriminator**: Classifies images as real or fake.

The project is implemented using TensorFlow and trained on the CelebA dataset.

## Dataset

The **[CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)** is used for training. It contains over 200,000 aligned and cropped images of celebrity faces.

## Project Features

1. **Data Preprocessing**:
   - Images are resized to 32x32.
   - Normalization is applied to scale pixel values between 0 and 1.

2. **GAN Architecture**:
   - **Generator**: Converts random noise vectors (latent space) into fake images.
   - **Discriminator**: Differentiates between real and fake images.

3. **Training**:
   - Alternating training of the generator and discriminator.
   - Loss functions: Binary cross-entropy for both networks.

4. **Visualization**:
   - 4x4 grid of generated images at different training steps.

## Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- Matplotlib
- NumPy

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-generation-using-gans.git
   cd face-generation-using-gans
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   - Download the CelebA dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
   - Extract the dataset and update the `directory` path in the notebook.

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Face Generation using GANs.ipynb
   ```

2. Follow the notebook instructions to preprocess data, define the model, and train the GAN.

3. View generated images during training in the notebook.

