from fastapi import (FastAPI, 
                     File, 
                     UploadFile, 
                     Request, 
                     HTTPException, 
                     Response)
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import uuid
import cv2
import numpy as np
import uvicorn
from fastapi.testclient import TestClient
import pathlib
from pathlib import Path
import sys
import platform
import base64



sys.path.insert(0, '/objdetection/app/yolov5')
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath




app = FastAPI()

CELLPHONEMODEL_PATH = Path(str('/objdetection/app/best_cellphone.pt'))
SEATBELTMODEL_PATH = Path(str('/objdetection/app/best.pt'))
YOLOV5MODEL = Path('/objdetection/app/yolov5')
client = TestClient(app)
IMAGEDIR = "/objdetection/app/fastapi-images/"
image_names = []
allowed_image_types = ["image/jpeg", "image/png", "image/gif"]
image_dict = {}


#load models from saved '.pt' path 
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model_cellphone = torch.hub.load(YOLOV5MODEL, 'custom', path=CELLPHONEMODEL_PATH, force_reload=True, source='local', device='cpu') 
model_seatbelt = torch.hub.load(YOLOV5MODEL, 'custom', path=SEATBELTMODEL_PATH, force_reload=True, source='local', device='cpu') 
model_cellphone.conf = 0.25
model_seatbelt.conf = 0.25

async def convert2cv2(pil_image):
    """
    converts PIL images
    to opencv images
    """
    image = pil_image.convert('RGB')
    opencvImage=np.array(image)
    if opencvImage is None or np.all(opencvImage==0):
        return np.zeros(shape=(150, 150, 3))
    opencvImage=cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB)
    return opencvImage

# async def encode_image_2base64(image):
#     buffered = io.BytesIO()
#     image.save(buffered, format='JPEG')
#     image_byte_arr = buffered.getvalue()
#     return base64.b64decode(image_byte_arr).decode("utf-8")

#write exceptions here:
class notReceivedException(Exception):
    def __init__(self, name: str):
        self.name = name
@app.exception_handler(notReceivedException)
async def notReceived_exception_handler(request: Request, exc:notReceivedException):
    return JSONResponse(
        status_code=404,
        content={"message": f"Oops! {exc.name} didn't receive!"},
    )


# @app.get("/")
# async def root_dir():
#     return {"msg": "Hi, this is root!"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), verbose:bool = False):
    """
    Detects both cellphones and seatbelts. 
    if you set the query parameter 'verbose' to True it will 
    return in this format rather than true or false:
    _____________________________________________________
    {   
        "class":string (cellphone, seatbelt ...) ,
        "confidence": float (how much confident my model is about the detected object. this parameter is between 0 and 1. 1->totaly confident and 0->not confident at all!),
        "bbox": [x1, y1, x2, y2] (list of four float parameters that shows the location of the detected object in the image)
    }
    _____________________________________________________
    if more than one object is detected then it will return 
    some dictionaries that each of them is associated 
    with one of the detected objects.
    In other words, number of returned dictionaries are the total number of detcted objects!
    """
    #manage input:
    if not file:
        raise notReceived_exception_handler(name = "file")
    if file.content_type not in allowed_image_types:
        raise HTTPException(status_code=400, 
                            detail="Invalid File type. (only jpeg, png or gif are allowed)"
                        )
    # detection = False

    #read images
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # image = image.rotate(180)

    #load models
    results_seatbelt = model_seatbelt(image)
    results_cellphone = model_cellphone(image)
    #if verbose is true then we must generate output in a particular format which i have explained above!
    if verbose:
        #generate file name and save it.
        file.filename = f"{uuid.uuid4()}"
        # image.save(f"{IMAGEDIR}{file.filename}.jpg")
        image_names.append(f"{file.filename}")
        image_dict[f"{file.filename}"] = [0, 0]
        # with Image.open(f"{IMAGEDIR}{file.filename}.jpg") as f:
        frame = np.array(image)
        #get the location of the detected object in the image:
        for box in results_cellphone.xyxy[0]: 
            if box[5]==0:
                # detection = True
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                rect = cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                # rect.save(f"{IMAGEDIR}{file.filename}_cellphone.jpg")
                im = Image.fromarray(rect)
                image_dict[f"{file.filename}"][0] =im
                # im.save(f"{IMAGEDIR}{file.filename}_cellphone.jpg")
        for box in results_seatbelt.xyxy[0]: 
            if box[5]==0:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                #draw a rectabgle around the detected object and save it somewhere:
                rect = cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                im = Image.fromarray(rect)
                image_dict[f"{file.filename}"][1] = im

                # im.save(f"{IMAGEDIR}{file.filename}_seatbelt.jpg")
            
    seatbelt_detections = []
    phone_detections = []
    for result in results_cellphone.xyxy[0]:
        x1, y1, x2, y2, conf, _ = result.tolist()
        phone_detections.append({
            "class": "cellphone",
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
        
    for result in results_seatbelt.xyxy[0]:
        x1, y1, x2, y2, conf, _ = result.tolist()
        seatbelt_detections.append({
            "class": "seatbelt",
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return {
        "phone detections": phone_detections if phone_detections and verbose else bool(phone_detections), 
        "seatbelt detections": seatbelt_detections if seatbelt_detections and verbose else bool(seatbelt_detections)
    }

@app.post("/cellphone")
async def detect_cellphone(file: UploadFile = File(...), verbose:bool=False):
    """
    This function merely returns cellphone detection outputs. 
    The verbose query variable works exactly like previous one. 
    if verbose is true, then instead of True or False it will return 
    a dictionary in the following format:
    _____________________________________________________
    {   
        "class":"cellphone",
        "confidence": float (how much confident my model is about the detected object. this parameter is between 0 and 1. 1->totaly confident and 0->not confident at all!),
        "bbox": [x1, y1, x2, y2] (list of four float parameters that shows the location of the detected object in the image which is cellphone in this case!)
    }
    _____________________________________________________

    """
    if not file:
        raise notReceived_exception_handler(name = "file")
    if file.content_type not in allowed_image_types:
        raise HTTPException(status_code=400, 
                            detail="Invalid File type. (only jpeg, png or gif are allowed)"
                        )
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # image = image.rotate(180)
    results_cellphone = model_cellphone(image)
    phone_detections = []
    for result in results_cellphone.xyxy[0]:
        x1, y1, x2, y2, conf, _ = result.tolist()
        phone_detections.append({
            "class": "cellphone",
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return {"phone detections": phone_detections if verbose else bool(phone_detections)}

@app.post("/seatbelt")
async def detect_seatbelt(file:UploadFile=File(...), verbose:bool = False):
    """
    This function merely returns seatbelt detection outputs. 
    The verbose query variable works exactly like previous one. 
    if verbose is true, then instead of True or False it will return 
    a dictionary in the following format:
    _____________________________________________________
    {   
        "class":"seatbelt",
        "confidence": float (how much confident my model is about the detected object. this parameter is between 0 and 1. 1->totaly confident and 0->not confident at all!),
        "bbox": [x1, y1, x2, y2] (list of four float parameters that shows the location of the detected object in the image which is seatbelt in this case!)
    }
    ______________________________________________________
    """
    if not file:
        raise notReceived_exception_handler(name = "file")
    if file.content_type not in allowed_image_types:
        raise HTTPException(status_code=400, 
                            detail="Invalid File type. (only jpeg, png or gif are allowed)"
                        )
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # image = image.rotate(180)
    results_seatbelt = model_seatbelt(image)
    seatbelt_detections = []
    for result in results_seatbelt.xyxy[0]:
        x1, y1, x2, y2, conf, _ = result.tolist()
        seatbelt_detections.append({
            "class": "seatbelt",
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
        
    return {"seatbelt detections": seatbelt_detections if verbose else bool(seatbelt_detections)}

@app.get("/display", responses = {
        200: {
            "content": {"image/png": {}}
        }
    },response_class=Response)
async def display_images():
    # print(image_dict)
    images = image_dict[image_names[-1]]
    # if not images[0]:
    #     raise notReceived_exception_handler(name = "file")
    print(type(images[0]))
    cvimg = await convert2cv2(images[0])
    res, im_png_cellphone = cv2.imencode(".jpg", cvimg)
    cvimg = await convert2cv2(images[1])
    res, im_png_seatbelt = cv2.imencode(".jpg", cvimg)
    return StreamingResponse(io.BytesIO(im_png_seatbelt.tobytes()), media_type="image/jpeg")
    # image_bytes_cell: bytes = Image.open(image[0])
    # image_bytes2: bytes = image_dict[image_names[-1]][1]
    # return {Response(content=image_bytes_cell, media_type="image/jpeg")}


if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=8000)

