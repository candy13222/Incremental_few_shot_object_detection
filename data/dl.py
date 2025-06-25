from pycocotools.coco import COCO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from tqdm.notebook import tqdm
import numpy as np


# instantiate COCO specifying the annotations json path
coco = COCO('/data3/age73423/ifsod/Deformable-DETR/data/coco/annotations/instances_train2017.json')
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=["truck"
,"traffic light"
,"fire hydrant"
,"stop sign"
,"parking meter"
,"bench"
,"elephant"
,"bear"
,"zebra"
,"giraffe"
,"backpack"
,"umbrella"
,"handbag"
,"tie"
,"suitcase"
,"frisbee"
,"skis"
,"snowboard"
,"sports ball"
,"kite"
,"baseball bat"
,"baseball glove"
,"skateboard"
,"surfboard"
,"tennis racket"
,"wine glass"
,"cup"
,"fork"
,"knife"
,"spoon"
,"bowl"
,"banana"
,"apple"
,"sandwich"
,"orange"
,"broccoli"
,"carrot"
,"hot dog"
,"pizza"
,"donut"
,"cake"
,"bed"
,"toilet"
,"laptop"
,"mouse"
,"remote"
,"keyboard"
,"cell phone"
,"microwave"
,"oven"
,"toaster"
,"sink"
,"refrigerator"
,"book"
,"clock"
,"vase"
,"scissors"
,"teddy bear"
,"hair drier"
,"toothbrush"
] )
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# handle annotations


ANNOTATIONS = {"info": {
    "description": "my-project-name"
}
}


def cocoJson(images: list) -> dict:
    arrayIds = np.array([k["id"] for k in images])
    annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for k in anns:
        k["category_id"] = catIds.index(k["category_id"])+1
    catS = [{'id': int(value), 'name': key}
            for key, value in categories.items()]
    ANNOTATIONS["images"] = images
    ANNOTATIONS["annotations"] = anns
    ANNOTATIONS["categories"] = catS

    return ANNOTATIONS


def createJson(JsonFile: json, label='train') -> None:
    name = label
    Path("data/labels").mkdir(parents=True, exist_ok=True)
    with open(f"data/labels/{name}.json", "w") as outfile:
        json.dump(JsonFile, outfile)

def downloadImages(images: list) -> None:
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    for im in tqdm(images):
        if not isfile(f"data/images/{im['file_name']}"):
            img_data = session.get(im['coco_url']).content
            with open('data/images/' + im['file_name'], 'wb') as handler:
                handler.write(img_data)


trainSet = cocoJson(images)
createJson(trainSet) 
downloadImages(images)