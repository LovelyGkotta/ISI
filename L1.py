import cv2
import numpy as np
from pdf_handle import PDFHandle


class ImageMatcher:
    def __init__(self, target_image_path):
        self.target_image = cv2.imread(target_image_path)

    def match_images(self, pdf_images):
        results = []
        for idx, image in enumerate(pdf_images):
            if self.target_image.shape != image.shape:
                continue

            difference = cv2.absdiff(self.target_image, image)
            if not np.any(difference):
                results.append((idx, image))

        return results

    def display_results(self, results):
        cv2.imshow(f'target image', self.target_image)
        for idx, image in results:
            cv2.imshow(f'Matched Image Page: {idx + 1}', image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pdf_path = "1912.11370v3.pdf"
    target_image_path = "images/287.png"
    extractor = PDFHandle(pdf_path)
    pdf_images = extractor.extract_images()


    # 1. 通过原图与源文件中获取到到图片进行最简单对比得到完全一致到图片
    # ⚠️：target image 是通过原图保存到图片，需要查找到对象在文件中一定是图片对象（PDF中存在大量图片分段，图片不是图片的问题）
    matcher = ImageMatcher(target_image_path)
    matches = matcher.match_images(pdf_images)
    matcher.display_results(matches)
