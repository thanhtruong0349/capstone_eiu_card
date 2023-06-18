from datetime import date
from pydantic import BaseModel

class Student(BaseModel):
    id = int
    s_id = str
    s_name = str    
    s_dob = str
    s_gender = str
    s_nationality = str
    s_major = str
    s_schoolyear = str
    s_description = str
    s_email = str
    s_phone = str
    
    class Config:
        orm_mode = True


# class Movie(BaseModel):
#     id = int
#     name = str
#     desc = str
#     type = str
#     url = str
#     rating = str
#     data = date
    
#     class Config:
#         orm_mode = True
