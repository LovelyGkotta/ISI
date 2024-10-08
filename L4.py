import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import faiss
from PIL import Image
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

target_image_path = "image_set/17330AFA-1BD8-42C4-AA36-7155650F2D56.png"

# 数据集向量化
class ImageVectorizer:
    def __init__(self):
        # Load pre-trained ResNet model for feature extraction
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove last classification layer
        self.model.eval()

        # Define transformation to resize image and convert it to tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def vectorize_image(self, image):
        # Convert OpenCV image (BGR) to PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply transformations and add batch dimension
        input_tensor = self.transform(pil_image).unsqueeze(0)

        # Extract features
        with torch.no_grad():
            vector = self.model(input_tensor).squeeze().numpy()

        return vector


# 使用向量搜索
class ImageSearchEngine:
    def __init__(self, vector_dim):
        # Initialize Faiss index for vector search (L2 distance metric)
        self.index = faiss.IndexFlatL2(vector_dim)
        self.vectors = []
        self.image_paths = []
        self.target_image = cv2.imread(target_image_path)

    def add_image(self, image_path, vector):
        # Add vector to index and store image path
        self.index.add(np.array([vector]))
        self.vectors.append(vector)
        self.image_paths.append(image_path)

    def search_similar_images(self, query_vector, top_k=5):
        # Search for top K similar vectors
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        return [(self.image_paths[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    def display_results(self, images_path):
        cv2.imshow("target_image", self.target_image)
        # 创建一个图形
        num_images = len(images_path)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        # 如果只有一张图像，则 axes 是单个对象而不是数组
        if num_images == 1:
            axes = [axes]

        for i, (image_path, distance) in enumerate(images_path):
            image = cv2.imread(image_path)

            # 将图像转换为RGB格式（matplotlib使用RGB，而OpenCV使用BGR）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 显示图像
            axes[i].imshow(image_rgb)
            axes[i].axis('off')  # 不显示坐标轴

            # 在图像上写上 idx 和 match_ratio
            text = f"distance: {distance:.3f}"
            axes[i].set_title(text, fontsize=12)

        plt.tight_layout()  # 自动调整子图间距
        plt.show()  # 显示图形
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize vectorizer and search engine
    vectorizer = ImageVectorizer()
    search_engine = ImageSearchEngine(vector_dim=2048)  # ResNet50 outputs 2048-dim vectors

    # Example: Adding images to the search engine
    image_set_path = "image_set"
    for filename in os.listdir(image_set_path):
        # Check if the file ends with .png
        if filename.endswith(".png"):
            image_path = os.path.join(image_set_path, filename)
            image = cv2.imread(image_path)
            vector = vectorizer.vectorize_image(image)
            search_engine.add_image(image_path, vector)

    # Example: Query with another image
    query_vector = vectorizer.vectorize_image(search_engine.target_image)

    # Search for similar images
    similar_images_path = search_engine.search_similar_images(query_vector, top_k=5)

    # display results
    search_engine.display_results(similar_images_path)

