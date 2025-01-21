# Face Generation using GANs

This project demonstrates the use of Generative Adversarial Networks (GANs) to generate human face images. The GAN consists of two neural networks:
- **Generator**: Creates fake images from random noise.
- **Discriminator**: Classifies images as real or fake.

The project is implemented using TensorFlow and trained on the CelebA dataset.

## Dataset

The **[CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)** is used for training. It contains over 200,000 aligned and cropped images of celebrity faces.

## Tech Stack

- **Programming Language**: Python
- **Frameworks and Libraries**: TensorFlow, Keras, NumPy, Matplotlib
- **Tools**: Jupyter Notebook, Kaggle

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

