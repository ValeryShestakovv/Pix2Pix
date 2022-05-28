from PIL import Image

def dataset_compile():
    img2 = Image.new("RGB", (2048, 1024), "white")
    for i in range(9900):
        img = Image.open(f"E:/LEARN/Pix2Pix-comics/data/input/{i}.jpg")
        img1 = Image.open(f"E:/LEARN/Pix2Pix-comics/data/target/{i}.jpg")

        img2.paste(img, (0, 0))

        img2.paste(img1, (1024, 0))

        img2.save(f"E:/LEARN/Pix2Pix-comics/data/train/{i}.jpg")

def rename():
    for i in range(999):
        img = Image.open(f"E:/LEARN/Pix2Pix-comics/data/train/{9900+i}.jpg")
        img.save(f"E:/LEARN/Pix2Pix-comics/data/val/{i}.jpg")

dataset_compile()