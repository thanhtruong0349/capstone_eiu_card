
from typing import Text
from sqlalchemy.schema import Column
from sqlalchemy.types import String, Integer, Text
from sources.Models.database import Base

class Student(Base):
    __tablename__ = "studentInfo"

    id = Column(Integer, primary_key=True, index=True)    
    s_id = Column(String(50), unique=True)
    s_name = Column(String(50))    
    s_dob = Column(String(20))
    s_gender = Column(String(20))
    s_nationality = Column(String(20))
    s_major = Column(String(20))
    s_schoolyear = Column(String(20))
    s_description = Column(String(100))
    s_email = Column(String(100))
    s_phone = Column(String(20))


# class Movie(Base):
#     __tablename__ = "Movie"

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(50), unique=True)
#     desc = Column(Text())
#     type = Column(String(20))
#     url = Column(String(100))
#     rating = Column(Integer)