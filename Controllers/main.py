import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from starlette.responses import RedirectResponse
from typing import Optional
from pylibsrtp import Session
from sources import app, templates
from fastapi import Request, UploadFile, File, Depends, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sources.Models import models
#from sources.Models.models import Feedback
from sources.Models.database import SessionLocal, engine, SQLALCHEMY_DATABASE_URL
import databases
from pydantic import BaseModel
import yolov5
import os
from PIL import Image
from sources.Controllers import utils
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import sources.Controllers.config as cfg
from sources.Controllers.student_card_detection import detect_card

from sources.Models.models import Student
# from sources.Models.models import Movie
import sources.Models.schema as schema
from sources.Models.database import SessionLocal, engine
#import model

""" ---- Setup ---- """
# Init Database
#database = databases.Database(SQLALCHEMY_DATABASE_URL)
models.Base.metadata.create_all(bind=engine)

def get_database_session():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# Startup database server before start app
# @app.on_event("startup")
# async def startup_database():
#     await database.connect()
    

# Shutdown database sever after closed app
# @app.on_event("shutdown")
# async def shutdown():
#     await database.disconnect()

# Init yolov5 model
CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)
FACE_MODEL = yolov5.load(cfg.FACE_MODEL_PATH)

# Set conf and iou threshold -> Remove overlap and low confident bounding boxes


CORNER_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CORNER_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR
FACE_CROP_DIR = cfg.FACE_DIR

""" ---- ##### -----"""
class feedback_Request(BaseModel):
    content: str
    rating: int
    class Config:
        orm_mode = True

class contact_Request(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    message: str
    class Config:
        orm_mode = True
    
""" Recognizion detected parts in ID """
config = Cfg.load_config_from_name('vgg_seq2seq') # OR vgg_transformer -> acc || vgg_seq2seq -> time
# config = Cfg.load_config_from_file(cfg.OCR_CFG)
# config['weights'] = cfg.OCR_MODEL_PATH
config['cnn']['pretrained']=False
config['device'] = cfg.DEVICE
config['predictor']['beamsearch']=False
detector = Predictor(config)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/id_card")
async def id_extract_page(request: Request):
    return templates.TemplateResponse("idcard.html", {"request": request})

@app.get("/ccic_card")
async def id_extract_page(request: Request):
    return templates.TemplateResponse("ccic_card.html", {"request": request})


@app.get("/ekyc")
async def ekyc_page(request: Request):
    return templates.TemplateResponse("ekyc.html", {"request": request})

@app.get("/feedback")
async def feedback_page(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})



@app.get("/contact")
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/result")
async def view_result_page(request: Request, db: Session = Depends(get_database_session)):
    records = db.query(Student).all()
    return templates.TemplateResponse("view_result.html", {"request": request})

@app.patch("/student/{id}")
async def update_movie(request: Request, id: int, db: Session = Depends(get_database_session)):
    requestBody = await request.json()
    student = db.query(Student).get(id)
    student.s_id = requestBody['s_id']
    student.s_name = requestBody['s_name']
    student.s_dob = requestBody['s_dob']
    student.s_gender = requestBody['s_gender']
    student.s_nationality = requestBody['s_nationality']
    student.s_major = requestBody['s_major']
    student.s_schoolyear = requestBody['s_schoolyear']
    student.s_description= requestBody['s_description']
    student.s_email = requestBody['s_email']
    student.s_phone = requestBody['s_phone']
    db.commit()
    db.refresh(student)
    newStudent = jsonable_encoder(student)
    return newStudent

@app.post("/student")
# @app.api_route("/extract", methods=["GET", "POST"])
async def savingNewStudent(db: Session = Depends(get_database_session),
                            studentName: schema.Student.s_name = Form(...),
                            studentID: schema.Student.s_id = Form(...), 
                            studentDoB: schema.Student.s_dob = Form(...), 
                            studentGender: schema.Student.s_gender = Form(...), 
                            studentNationality: schema.Student.s_nationality = Form(...), 
                            studentMajor: schema.Student.s_major = Form(...), 
                            studentSchoolyear: schema.Student.s_schoolyear = Form(...), 
                            studentDescription: schema.Student.s_description = Form(...),
                            studentEmail: schema.Student.s_email = Form(...),
                            studentPhone: schema.Student.s_phone = Form(...)):
 
    item = Student(s_id=studentID, s_name=studentName,  
                   s_dob=studentDoB, s_gender=studentGender, 
                   s_nationality=studentNationality, s_major=studentMajor, 
                   s_schoolyear=studentSchoolyear, s_description=studentDescription, 
                   s_email=studentEmail, s_phone=studentPhone)
    db.add(item)
    db.commit()
    db.refresh(item)

    response = RedirectResponse('/result', status_code=303)
    # print("[INFO] add data to database")
    return response
@app.patch("/createStudentItem/{id}")
async def update_student(request: Request, id: int, db: Session = Depends(get_database_session)):
    requestBody = await request.json()
    student = db.query(Student).get(id)
    student.s_id = requestBody['s_id']
    student.s_name = requestBody['s_name']
    student.s_dob = requestBody['s_dob']
    student.s_gender = requestBody['s_gender']
    student.s_nationality = requestBody['s_nationality']
    student.s_major = requestBody['s_major']
    student.s_schoolyear = requestBody['s_schoolyear']
    student.s_description= requestBody['s_description']
    student.s_email = requestBody['s_email']
    student.s_phone = requestBody['s_phone']
    db.commit()
    db.refresh(student)
    newStudent = jsonable_encoder(student)
    return student

@app.post("/createStudentItem")
# @app.api_route("/extract", methods=["GET", "POST"])
async def createStudentItem(db: Session = Depends(get_database_session),
                            studentName: schema.Student.s_name = Form(...),
                            studentID: schema.Student.s_id = Form(...), 
                            studentDoB: schema.Student.s_dob = Form(...), 
                            studentGender: schema.Student.s_gender = Form(...), 
                            studentNationality: schema.Student.s_nationality = Form(...), 
                            studentMajor: schema.Student.s_major = Form(...), 
                            studentSchoolyear: schema.Student.s_schoolyear = Form(...), 
                            studentDescription: schema.Student.s_description = Form(...),
                            studentEmail: schema.Student.s_email = Form(...),
                            studentPhone: schema.Student.s_phone = Form(...)):
    # add data to database 
    # print("[INFO] get data from Form")
    # name = "a5"
    # desc = "a"
    # type = "movies"
    # url = "b"
    # rate = "90"
    item = Student(s_id=studentID, s_name=studentName,  
                   s_dob=studentDoB, s_gender=studentGender, 
                   s_nationality=studentNationality, s_major=studentMajor, 
                   s_schoolyear=studentSchoolyear, s_description=studentDescription, 
                   s_email=studentEmail, s_phone=studentPhone)
    db.add(item)
    db.commit()
    db.refresh(item)

    response = RedirectResponse('/result', status_code=303)
    # print("[INFO] add data to database")
    return response

# @app.post("/movie/")
# async def create_movie(db: Session = Depends(get_database_session), name: schema.Movie.name = Form(...), url: schema.Movie.url = Form(...), rate: schema.Movie.rating = Form(...), type: schema.Movie.type = Form(...), desc: schema.Movie.desc = Form(...)):
#     movie = Movie(name=name, url=url, rating=rate, type=type, desc=desc)
#     db.add(movie)
#     db.commit()
#     db.refresh(movie)
#     response = RedirectResponse('/id_card', status_code=303)
#     print("[INFO] add data to database")
#     return response

@app.post("/uploader")
async def upload(file: UploadFile = File(...)):
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        for uploaded_img in INPUT_IMG:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))

    file_location = f"./{UPLOAD_FOLDER}/{file.filename}"
    contents = await file.read()
    with open(file_location, 'wb') as f:
        f.write(contents)
    
    # Validating file
    INPUT_FILE = os.listdir(UPLOAD_FOLDER)[0]
    if INPUT_FILE == 'NULL':
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        error = "No file selected!"
        return JSONResponse(status_code=403, content={"message": error})
    elif INPUT_FILE == 'WRONG_EXTS':
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        error = "This file is not supported!"
        return JSONResponse(status_code=404, content={"message": error})

    # return {"Filename": file.filename}
    return await extract_student_card()

@app.post("/extract")
# @app.api_route("/extract", methods=["GET", "POST"])
async def extract_info(ekyc=False, path_id = None):
    """ Check if uploaded image exist """
    if not os.path.isdir(cfg.UPLOAD_FOLDER):
        os.mkdir(cfg.UPLOAD_FOLDER)
        
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        if not ekyc:
            img = os.path.join(UPLOAD_FOLDER, INPUT_IMG[0])
        else:
            img = path_id

    CORNER = CORNER_MODEL(img)
    # CORNER.save(save_dir='results/')
    predictions = CORNER.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if len(categories) != 4:
        error = "Detecting corner failed!"
        return JSONResponse(status_code=401, content={"message": error})
    boxes = utils.class_Order(predictions[:, :4].tolist(), categories) # x1, x2, y1, y2
    IMG = Image.open(img)
    center_points = list(map(utils.get_center_point, boxes))

    """ Temporary fixing """
    c2, c3 = center_points[2], center_points[3]
    c2_fix, c3_fix = (c2[0],c2[1]+30), (c3[0],c3[1]+30)
    center_points = [center_points[0], center_points[1], c2_fix, c3_fix]
    center_points = np.asarray(center_points)
    aligned = utils.four_point_transform(IMG, center_points)
    # Convert from OpenCV to PIL format
    aligned = Image.fromarray(aligned)
    # aligned.save('res.jpg')
    # CORNER.show()

    CONTENT = CONTENT_MODEL(aligned)
    # CONTENT.save(save_dir='results/')
    predictions = CONTENT.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if 7 not in categories:
        if len(categories) < 9:
            error = "Missing fields! Detecting content failed!"
            return JSONResponse(status_code=402, content={"message": error})
    elif 7 in categories:
        if len(categories) < 10:
            error = "Missing fields! Detecting content failed!"
            return JSONResponse(status_code=402, content={"message": error})

    boxes = predictions[:,:4].tolist()

    """ Non Maximum Suppression """
    boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)
    boxes = utils.class_Order(boxes, categories) # x1, x2, y1, y2
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    else:
        for f in os.listdir(SAVE_DIR):
            os.remove(os.path.join(SAVE_DIR, f))

    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        if 5 < index < 9:
            # right = c3[0] 
            right = right + 100
        cropped_image = aligned.crop((left,top,right,bottom))
        cropped_image.save(os.path.join(SAVE_DIR, f'{index}.jpg'))

    FIELDS_DETECTED = [] # Collecting all detected parts
    for idx, img_crop in enumerate(sorted(os.listdir(SAVE_DIR))):
        if idx > 0:
            img_ = Image.open(os.path.join(SAVE_DIR,img_crop))
            s = detector.predict(img_)
            FIELDS_DETECTED.append(s)

    if 7 in categories:
        FIELDS_DETECTED = FIELDS_DETECTED[:6] + [FIELDS_DETECTED[6] + ', ' + FIELDS_DETECTED[7]] + [FIELDS_DETECTED[8]]

    response = {
                "data": FIELDS_DETECTED
            }

    response = jsonable_encoder(response)
    return JSONResponse(content=response) 

@app.get("/extract_student_card")
async def extract_student_card(ekyc=False, path_id = None):
    
    """ Check if uploaded image exist """
    if not os.path.isdir(cfg.UPLOAD_FOLDER):
        os.mkdir(cfg.UPLOAD_FOLDER)
        
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        
        img = os.path.join(UPLOAD_FOLDER, INPUT_IMG[0])
       
    
    # delete old photos before              
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    else:
        for f in os.listdir(SAVE_DIR):
            os.remove(os.path.join(SAVE_DIR, f))
    #  ===================================
    # This part for student card detection 
    # detect card type
    STUDENT_MODEL_PATH = cfg.STUDENT_MODEL_PATH
    frame = detect_card(img, STUDENT_MODEL_PATH)

    FIELDS_DETECTED = [None] * 9 # Collecting all detected parts

    for idx, img_crop in enumerate(sorted(os.listdir(SAVE_DIR))):
        if idx > 0 and idx < 9:
            img_ = Image.open(os.path.join(SAVE_DIR,img_crop))
            s = detector.predict(img_)
            # FIELDS_DETECTED.append(s)
            img_id = os.path.splitext(img_crop)[0]
            FIELDS_DETECTED[int(img_id)] = s
            print(FIELDS_DETECTED)
            # print(os.path.splitext(img_crop)[0])

    # if 7 in categories:
    #     FIELDS_DETECTED = FIELDS_DETECTED[:6] + [FIELDS_DETECTED[6] + ', ' + FIELDS_DETECTED[7]] + [FIELDS_DETECTED[8]]
    FIELDS_DETECTED[0] = "Card"
    FIELDS_DETECTED[6] = "Photo"
    print("[INFO] FIELD value: {}".format(FIELDS_DETECTED))
    # # add data to database 
    # name = "a3"
    # desc = "a"
    # type = "movies"
    # url = "b"
    # rate = "90"
    # movie = Movie(name=name, url=url, rating=rate, type=type, desc=desc)
    # db.add(movie)
    # db.commit()
    # db.refresh(movie)
    # #response = RedirectResponse('/movie', status_code=303)
    # print("[INFO] add data to database")

    response = {"data": FIELDS_DETECTED}
    response = jsonable_encoder(response)    
    return JSONResponse(content=response)


@app.post("/download")
async def download(file: str = Form(...)):
    if file != 'undefined':
        noti = 'Download file successfully!'
        return JSONResponse(status_code=201, content={"message": noti})
    else:
        error = 'No file to download!'
        return JSONResponse(status_code=405, content={"message": error})

# @app.post("/feedback")
# async def save_feedback(content: str = Form(...), rating: int = Form(...), db: Session = Depends(get_db)):
#     feedback = Feedback()
#     feedback.content = content
#     feedback.rating = rating
#     db.add(feedback)
#     db.commit()

#     response = {
#                 "code": "200",
#                 "content": "save successfully" 
#                 }
    
#     return JSONResponse(content=response)

@app.post("/contact")
async def contact(request: contact_Request):
    # print(request.name)
    pass

@app.post("/ekyc/uploader")
async def get_id_card(id: UploadFile = File(...), img: UploadFile = File(...)):
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        for uploaded_img in INPUT_IMG:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))

    id_location = f"./{UPLOAD_FOLDER}/{id.filename}"
    id_contents = await id.read()

    with open(id_location, 'wb') as f:
        f.write(id_contents)

    img_location = f"./{UPLOAD_FOLDER}/{img.filename}"
    img_contents = await img.read()
    with open(img_location, 'wb') as f_:
        f_.write(img_contents)
    
    # Validating file
    INPUT_FILE = os.listdir(UPLOAD_FOLDER)
    if 'NULL_1' in INPUT_FILE and 'NULL_2' not in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing ID card image!"
        return JSONResponse(status_code=410, content={"message": error})
    elif 'NULL_2' in INPUT_FILE and 'NULL_1' not in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing person image!"
        return JSONResponse(status_code=411, content={"message": error})
    elif 'NULL_1'in INPUT_FILE and 'NULL_2' in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing ID card and person images!"
        return JSONResponse(status_code=412, content={"message": error})
    else:
        id_name = id.filename.split('.')
        new_id_name = f"./{UPLOAD_FOLDER}/id.{id_name[-1]}"
        os.rename(id_location, new_id_name)
        img_name = img.filename.split('.')
        new_img_name = f"./{UPLOAD_FOLDER}/person.{img_name[-1]}"
        os.rename(img_location, new_img_name)

    FACE = FACE_MODEL(new_img_name)
    predictions = FACE.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if 0 not in categories:
        error = "No face detected!"
        return JSONResponse(status_code=413, content={"message": error})
    elif categories.count(0) > 1:
        error = "Multiple faces detected!"
        return JSONResponse(status_code=414, content={"message": error})

    boxes = predictions[:,:4].tolist()

    """ Non Maximum Suppression """
    # boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)
    
    if not os.path.isdir(FACE_CROP_DIR):
        os.mkdir(FACE_CROP_DIR)
    else:
        for f in os.listdir(FACE_CROP_DIR):
            os.remove(os.path.join(FACE_CROP_DIR, f))
    
    FACE_IMG = Image.open(new_img_name)
    # left, top, right, bottom = boxes[0]
    cropped_image = FACE_IMG.crop((boxes[0]))
    cropped_image.save(os.path.join(FACE_CROP_DIR, 'face_crop.jpg'))

    return await extract_info(ekyc=True, path_id=new_id_name)