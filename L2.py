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
            for match_pair in matches:
                if len(match_pair) == 2:  # 确保有两个匹配
                    m, n = match_pair
                    if m.distance < 0.4 * n.distance:  # 使用比率测试筛选匹配
                        good_matches.append(m)

            # 画出匹配结果
            match_img = cv2.drawMatches(
                target_gray, kp_target, image, kp_image,
                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # 显示结果
            # cv2.imshow(f"Matches for Image {idx + 1}", match_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 计算匹配度
            match_ratio = len(good_matches) / len(kp_target) if len(kp_target) > 0 else 0
            results.append((match_ratio, idx, image))

            results.sort(reverse=True, key=lambda x: x[0])  # 根据匹配比例降序排序
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
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 显示图像
            axes[i].imshow(image_rgb)
            axes[i].axis('off')  # 不显示坐标轴
            text = f"Page: {idx + 1}, Ratio: {match_ratio:.2f}"
            axes[i].set_title(text, fontsize=12)

        plt.tight_layout()  # 自动调整子图间距
        plt.show()  # 显示图形
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pdf_path = "1912.11370v3.pdf"
    target_image_path = "Screenshot 2024-10-07 at 07.02.33.png"
    extractor = PDFHandle(pdf_path)
    pdf_images = extractor.extract_images()

    # 2. 通过截图的图片（相较于原图，有精度损失，尺寸不同，多余边缘等问题）来与源文件中获取到到图片进行SIFT特征点对比得到像素级别相似的图片
    # Notice:   1. 由于SIFT是通过特征点匹配的图片，确认特征点需要的领域范围为3σ(d+1)*3σ(d+1)所以小图片（32*32以下）的图片可能无法找到特征点，
    #           2. 虽然泛用性变高了，但是带了的就是如果出现图标等较为相似的图像时候可能会出错
    #           3. 仍旧需要保证图片是图片！
    matcher = ImageMatcher(target_image_path)
    matches = matcher.match_images(pdf_images)
    matcher.display_results(matches)
