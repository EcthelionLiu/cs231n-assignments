from tensorflow.python.framework.ops import device_v2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].
    """
    similarity = None
    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################
    text_norm = text_features / torch.linalg.norm(text_features, axis = 1, keepdims = True)
    image_norm = image_features / torch.linalg.norm(image_features, axis = 1, keepdims = True)
    similarity = text_norm @ image_norm.T
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################
    text_tokens = clip.tokenize(class_texts).to(device)
    text_features = clip_model.encode_text(text_tokens)
    
    processed_images = [clip_preprocess(Image.fromarray(img)).unsqueeze(0) for img in images ]
    images_tensor = torch.cat(processed_images, dim=0).to(device)
    image_features = clip_model.encode_image(images_tensor)
    
    similarity = get_similarity_no_loop(text_features, image_features)
    pred_classes = [class_texts[i] for i in torch.argmax(similarity, axis = 0)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        self.clip_model = clip_model
        self.device = device
        
        processed_images = [clip_preprocess(Image.fromarray(img)).unsqueeze(0) for img in images ]
        images_tensor = torch.cat(processed_images, dim=0).to(device)
        self.image_features = clip_model.encode_image(images_tensor)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################
        text_tokens = clip.tokenize([query]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        
        similarity = get_similarity_no_loop(text_features, self.image_features)
        topk = torch.Tensor.topk(similarity, k = k, dim = 1)
        top_indices = topk.indices.cpu().tolist()[0]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return top_indices

  
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import cv2
from glob import glob
from torchvision.transforms.functional import pil_to_tensor

def create_segmentation_overlay(mask, frame, alpha=0.5):
    color_mask = np.zeros_like(frame)
    color_mask[:, :, 1] = mask * 255  # 绿色区域
    return cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

class DavisDataset:
    def __init__(self, root_dir="/path/to/DAVIS", split="val"):
        """
        root_dir: DAVIS 数据集根目录（包含 'Annotations'、'JPEGImages' 等）
        split: 'train', 'val' 或 'trainval'
        """
        self.img_root = os.path.join(root_dir, "JPEGImages", "480p")
        self.mask_root = os.path.join(root_dir, "Annotations", "480p")

        image_sets_path = os.path.join(root_dir, "ImageSets", "2017", f"{split}.txt")
        with open(image_sets_path) as f:
            self.video_names = [x.strip() for x in f.readlines() if x.strip()]
        
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), 
            T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])

    def __len__(self):
        return len(self.video_names)

    def get_sample(self, index, merge_instances=True):
        video_name = self.video_names[index]
        img_dir = os.path.join(self.img_root, video_name)
        mask_dir = os.path.join(self.mask_root, video_name)

        img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))

        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in img_files]
        masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]

        masks = np.array(masks)
        if merge_instances:
            # 合并相同实例编号，以便语义连续编号
            unique_ids = np.unique(masks)
            id_map = {old: new for new, old in enumerate(unique_ids)}
            masks = np.vectorize(id_map.get)(masks)

        num_classes = int(masks.max() + 1)
        print(f"video {video_name}: {len(frames)} frames, num_classes={num_classes}")
        return np.array(frames), masks, num_classes

    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
                tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])
        return torch.stack(res)

    def process_masks(self, masks, device):
        # 强制转换成 list -> 再组合成正规数组
        if isinstance(masks, np.ndarray) and masks.dtype == object:
            masks = list(masks)
    
        # 保证每个元素都是二维 ndarray
        clean_masks = []
        for i, m in enumerate(masks):
            if isinstance(m, list):
                m = np.array(m)
            if m is None:
                raise ValueError(f"Mask #{i} is None — check missing file.")
            if m.ndim == 3:
                m = m.squeeze()
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            clean_masks.append(m)
    
        masks = np.stack(clean_masks)  # (N, H, W)
    
        res = []
        for m in masks:
            m = cv2.resize(m, (60, 60), interpolation=cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten())
        return torch.stack(res).to(device)

    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy().reshape((60, 60))
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"
    n = int(segmentation_mask.max()) + 1
    # Generate deterministic colors for each class using a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(n)

    # Create a color image for the segmentation mask
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU)."""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################
        self.device = device
        self.num_classes = num_classes
        self.inp_dim = inp_dim
        
        self.model = nn.Sequential(
            nn.Linear(inp_dim, inp_dim // 2),
            nn.BatchNorm1d(inp_dim // 2),
            nn.GELU(),
            nn.Linear(inp_dim // 2, num_classes),
        ).to(device)
        
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        for _ in (pbar := tqdm(range(num_iters), desc="Training")):
            self.optim.zero_grad()
            X_pred = self.model(X_train)
            loss = self.loss_fn(X_pred, Y_train)
            loss.backward()
            self.optim.step()

            pbar.set_postfix(loss=loss.item())
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.
        """
        pred_classes = None
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        pred_classes = torch.argmax(self.model(X_test), dim = 1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return pred_classes