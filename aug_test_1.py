import cv2
import albumentations as A 
import os
import shutil
from tqdm import tqdm


INPUT_DIR = "yolo_test_1"
OUTPUT_DIR = "augmented_data"

IMG_DIR = os.path.join(INPUT_DIR, "images")
LBL_DIR = os.path.join(INPUT_DIR, "labels")

OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

shutil.copy(os.path.join(INPUT_DIR, "classes.txt"), OUTPUT_DIR)



transform = A.Compose(
    [
        # geometrik dönüşümler
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.4),
        A.Rotate(limit=20, p=0.5),

        # hava durumu
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_length=15, p=1),
            A.RandomSnow(snow_point_range=(0.1, 0.3), brightness_coeff=1.5, p=1),
            A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.08, p=1)
        ], p=0.3),

        #kamera arıza ve bozulmaları
        A.OneOf([
            A.Blur(blur_limit=5, p=1),
            A.MotionBlur(blur_limit=5, p=1),
            A.GaussNoise(p=0.2),
        ], p=0.3),

        #ölü piksel
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(5, 8),
            hole_width_range=(5, 8),
            fill=0,
            p=0.2
        ),

        # ışık ve renk
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),

        # rgb/thermal
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3 # nesnenin %30u görünüyorsa etiket kalsın
    )
)



def load_labels(label_path):
    bboxes = []
    class_labels = []

    if not os.path.exists(label_path):
        return bboxes, class_labels
    
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

    return bboxes, class_labels


def save_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for bbox, cls in zip(bboxes, class_labels):
            x, y, w, h = bbox
            f.write(f"{cls} {x} {y} {w} {h}\n")


AUGMENT_COUNT = 3 # her resimden kaç tane üretilecek

image_files = os.listdir(IMG_DIR)

for img_name in tqdm(image_files):
    img_path = os.path.join(IMG_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LBL_DIR, base_name + ".txt")

    image = cv2.imread(img_path)
    if image is None:
        continue

    bboxes, class_labels = load_labels(label_path)
    if len(bboxes) == 0: # etiketsiz fotoyu augmente yapma ama orijinali kopyala
        shutil.copy(img_path, os.path.join(OUT_IMG_DIR, img_name))
        continue

    shutil.copy(img_path, os.path.join(OUT_IMG_DIR, img_name))

    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(OUT_LBL_DIR, os.path.basename(label_path)))


    for i in range(AUGMENT_COUNT):
        augmented = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
        new_lbl_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.txt"

        cv2.imwrite(os.path.join(OUT_IMG_DIR, new_img_name), aug_img)
        save_labels(os.path.join(OUT_LBL_DIR, new_lbl_name), aug_bboxes, aug_labels)


print("Augmentation tamamlandı!")








