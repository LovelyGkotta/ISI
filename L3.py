import cv2
import matplotlib.pyplot as plt
from pdf_handle import PDFHandle


class ImageMatcher:
    def __init__(self, target_image_path):
        self.target_image = cv2.imread(target_image_path)

    def match_images(self, pdf_images):
        sift = cv2.SIFT_create()

        # 提取目标图像的关键点和描述符
        target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        kp_target, des_target = sift.detectAndCompute(target_gray, None)

        results = []

        for idx, image in enumerate(pdf_images):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 提取当前图像的关键点和描述符
            kp_image, des_image = sift.detectAndCompute(image_gray, None)

            # 检查描述符是否为 None
            if des_image is None or des_target is None:
                print(f"No descriptors found for image {idx}, skipping...")
                continue

            # 使用 KNN 进行匹配
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des_target, des_image, k=2)

            # 筛选匹配项
            good_matches = []
            used_target_indices = set()  # 记录已匹配的目标图像特征点索引
            used_image_indices = set()    # 记录已匹配的当前图像特征点索引

            for match_pair in matches:
                if len(match_pair) == 2:  # 确保有两个匹配
                    m, n = match_pair
                    if m.distance < 0.6 * n.distance:  # 使用比率测试筛选匹配
                        if m.queryIdx not in used_target_indices and m.trainIdx not in used_image_indices:
                            good_matches.append(m)
                            used_target_indices.add(m.queryIdx)
                            used_image_indices.add(m.trainIdx)

            # 画出匹配结果
            match_img = cv2.drawMatches(
                self.target_image, kp_target, image, kp_image,
                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # 显示结果
            cv2.imshow(f"Matches for Image {idx + 1}", match_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 计算匹配度
            match_ratio = len(good_matches) / len(kp_target) if len(kp_target) > 0 else 0
            results.append((match_ratio, idx, image))

        # 根据匹配比例降序排序
        results.sort(reverse=True, key=lambda x: x[0])
        top_results = results[:5]  # 获取前五个结果

        return top_results

    def display_results(self, results):
        cv2.imshow("target_image", self.target_image)
        # 创建一个图形
        num_images = len(results)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        # 如果只有一张图像，则 axes 是单个对象而不是数组
        if num_images == 1:
            axes = [axes]

        for i, (match_ratio, idx, image) in enumerate(results):
            # 将图像转换为RGB格式（matplotlib使用RGB，而OpenCV使用BGR）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 显示图像
            axes[i].imshow(image_rgb)
            axes[i].axis('off')  # 不显示坐标轴

            # 在图像上写上 idx 和 match_ratio
            text = f"Page: {idx + 1}, Ratio: {match_ratio:.2f}"
            axes[i].set_title(text, fontsize=12)

        plt.tight_layout()  # 自动调整子图间距
        plt.show()  # 显示图形
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pdf_path = "1912.11370v3.pdf"
    target_image_path = "Screenshot 2024-09-28 at 23.06.18 copy.png"
    extractor = PDFHandle(pdf_path)
    pdf_page_images = extractor.extract_page_as_image()

    matcher = ImageMatcher(target_image_path)
    matches = matcher.match_images(pdf_page_images)
    matcher.display_results(matches)
