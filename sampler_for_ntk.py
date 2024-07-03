import numpy as np
from PIL import Image
from random import sample
from pathlib import Path
from argparse import ArgumentParser
from torchvision import transforms as T


def get_args():
    parser = ArgumentParser()
    parser.add_argument( "-r", "--root_dir", type=str, default="../data/ImageNet10/train", help="The path to the root directory of the dataset", )
    parser.add_argument( "-s", "--size", type=int, default=224, help="The size of the image" )
    parser.add_argument( "-n", "--num_sample", type=int, default=100, help="The number of samples to read from the dataset", )
    return parser.parse_args()


def get_num_sample_per_class(n_class, n_sample):
    if n_sample % n_class != 0:
        raise ValueError(
            "The number of samples must be divisible by the number of classes"
        )
    return n_sample // n_class


def get_transform(size):
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((size, size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.ToPILImage(),
        ]
    )


def get_sample(class_dir, n_sample_per_class):
    image_list = list(class_dir.glob("*.jpg"))
    sample_list = sample(image_list, n_sample_per_class)
    sample_list = [Image.open(img).convert('RGB') for img in sample_list]
    return sample_list


def main():

    args = get_args()
    root = Path(args.root_dir)
    n_class = len(list(root.iterdir()))  # In our case, 10
    n_sample_per_class = get_num_sample_per_class(n_class, args.num_sample)  # In our case, 50
    transform = get_transform(args.size)  # In our case, resize to 224 x 224

    image_list = []
    label_list = []

    for class_dir in root.iterdir():
        samples = get_sample(class_dir, n_sample_per_class)  # Sample 50 images from each class
        samples = [ np.array(transform(img)) for img in samples ]  # Do the preprocessing

        image_list.extend(samples)
        label_list.extend([int(class_dir.name)] * n_sample_per_class)
        
        
    image_array = np.stack(image_list, axis=0)
    label_array = np.array(label_list)

    np.save("IN10_image.npy", image_array)
    print(f"Save: IN10_image.npy\tShape: {image_array.shape}")

    np.save("IN10_label.npy", label_array)
    print(f"Save: IN10_label.npy\tShape: {label_array.shape}")


if __name__ == "__main__":
    main()
