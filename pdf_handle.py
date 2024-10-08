import numpy as np
import pdfplumber

class PDFHandle:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_images(self):
        images = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):

                if page.images:
                    print(f"Processing page {i + 1}/{len(pdf.pages)}...")

                    for img in page.images:
                        # Extracting image using the x0, y0, width, height
                        x0, y0 = img["x0"], img["top"]
                        x1, y1 = img["x1"], img["bottom"]
                        image_bbox = (x0, y0, x1, y1)
                        cropped_image = page.within_bbox(image_bbox).to_image()
                        pil_img = cropped_image.original.convert("RGB")
                        np_img = np.array(pil_img)[:, :, ::-1]  # Convert to BGR format for OpenCV
                        images.append(np_img)
                        # cv2.imshow(f"./{len(images)}", np_img)
                        # cv2.waitKey()
                        # cv2.imwrite(f"images/{len(images)}.png", np_img)
        return images

    def extract_page_as_image(self):
        page_images = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Convert the page to an image (resolution in DPI, e.g., 72 or 150)
                pil_img = page.to_image(resolution=150).original
                np_img = np.array(pil_img)[:, :, ::-1]  # Convert to BGR format for OpenCV
                page_images.append(np_img)

        return page_images