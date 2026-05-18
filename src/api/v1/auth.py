from fastapi import APIRouter
from src.pydantic_models.auth import User
from src.utils.util import encrypt_password, verify_password, generate_token


router = APIRouter()
user_db = {}


@router.post("/register")
def register(user:User):
   """
   1. generate hashed password
   2. store creds in db
   """
   password = user.password
   password_hash = encrypt_password(password=password)
   user.password = password_hash
   user_db[user.email] = user.model_dump()
   print(f"registered user: {user_db}")
   return True


@router.post("/login")
def login(user:User):
   """
      1. Fetch hashed password from db for the given username
      2. compare hashed and plain password using bcrypt
   """
   # fetch plain password from db
   print(f"user:{user}")
   print(f"user db : {user_db}")
   hash_pw = user_db.get(user.email).get("password")
   print(f"user: {user}")
   if not verify_password(hash_pw=hash_pw ,password=user.password):
      print(f"Invalid credentials")
      return
   token = generate_token(payload=user.model_dump())
   return token
