from pydantic import BaseModel

class User(BaseModel):
    fullname:str = None
    email: str = "email@example.com"
    password: str = "password"

class Query(BaseModel):
   question:str