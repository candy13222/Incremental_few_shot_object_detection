# from pycocotools.coco import COCO

# dataDir='/data3/age73423/ifsod/Deformable-DETR/data/coco'
# dataType='train2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# # initialize COCO api for instance annotations
# coco=COCO(annFile)

# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# cat_nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))
# cat_nums = ["truck"
# ,"traffic light"
# ,"fire hydrant"
# ,"stop sign"
# ,"parking meter"
# ,"bench"
# ,"elephant"
# ,"bear"
# ,"zebra"
# ,"giraffe"
# ,"backpack"
# ,"umbrella"
# ,"handbag"
# ,"tie"
# ,"suitcase"
# ,"frisbee"
# ,"skis"
# ,"snowboard"
# ,"sports ball"
# ,"kite"
# ,"baseball bat"
# ,"baseball glove"
# ,"skateboard"
# ,"surfboard"
# ,"tennis racket"
# ,"wine glass"
# ,"cup"
# ,"fork"
# ,"knife"
# ,"spoon"
# ,"bowl"
# ,"banana"
# ,"apple"
# ,"sandwich"
# ,"orange"
# ,"broccoli"
# ,"carrot"
# ,"hot dog"
# ,"pizza"
# ,"donut"
# ,"cake"
# ,"bed"
# ,"toilet"
# ,"laptop"
# ,"mouse"
# ,"remote"
# ,"keyboard"
# ,"cell phone"
# ,"microwave"
# ,"oven"
# ,"toaster"
# ,"sink"
# ,"refrigerator"
# ,"book"
# ,"clock"
# ,"vase"
# ,"scissors"
# ,"teddy bear"
# ,"hair drier"
# ,"toothbrush"
# ] 

# # 统计各类的图片数量和标注框数量
# for cat_name in cat_nms:
#     catId = coco.getCatIds(catNms=cat_name)
#     imgId = coco.getImgIds(catIds=catId)
#     annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)

#     print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))
import json
import random
ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
CLASS2ID = {v: k for k, v in ID2CLASS.items()}
data_path = "coco/cocosplit/datasplit/trainvalno5k.json"
data = json.load(open(data_path))

new_all_cats = []
for cat in data["categories"]:
    new_all_cats.append(cat)
id2img = {}
for i in data["images"]:
    id2img[i["id"]] = i


anno = {i: [] for i in ID2CLASS.keys()}
for a in data["annotations"]:
    if a["iscrowd"] == 1:
        continue
    anno[a["category_id"]].append(a)

for i in range(0,10):
    random.seed(i)
    for c in ID2CLASS.keys():
        img_ids = {}
        for a in anno[c]:
            if a["image_id"] in img_ids:
                img_ids[a["image_id"]].append(a)
            else:
                img_ids[a["image_id"]] = [a]

        sample_shots = []
        sample_imgs = []

        for shots in [1]:
            while True:
                imgs = random.sample(list(img_ids.keys()), shots)
                print(imgs)
                for img in imgs:
                    skip = False
                    for s in sample_shots:
                        if img == s["image_id"]:
                            skip = True
                            break
                    if skip:
                        continue
                    if len(img_ids[img]) + len(sample_shots) > shots:
                        continue
                    sample_shots.extend(img_ids[img])
                    sample_imgs.append(id2img[img])
                    if len(sample_shots) == shots:
                        break
                if len(sample_shots) == shots:
                    break
            new_data = {
                "info": data["info"],
                "licenses": data["licenses"],
                "images": sample_imgs,
                "annotations": sample_shots,
            }
            # save_path = get_save_path_seeds(
            #     data_path, ID2CLASS[c], shots, i
            # )
            # new_data["categories"] = new_all_cats
            # with open(save_path, "w") as f:
            #     json.dump(new_data, f)