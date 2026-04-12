"""
function to:
generate token.
login comparison
docorator to validate token
"""
import jwt
import bcrypt
from functools import wraps
from fastapi import HTTPException, Request
from src.auth.models import User

user = User()
hash_pw = b'$2b$12$C1Ma5newVXzYm/Q1vOTHiOwLD5kYo/TEVVmiYgSjlKNWrw84L8Z4q'
def generate_token(payload):
    payload = user.model_dump()
    token = jwt.encode(payload=payload, key="secret", algorithm="HS256")
    print(f"token generated successfully: {token}")
    return token

def verify_password(hash_pw, password):
    password_bytes = password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hash_pw)

def token_validation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        if not request:
            raise HTTPException(status_code=400, detail="Request missing")
        
        token = request.headers.get("Authorization").split(" ")[-1]
        if not token:
            raise HTTPException(status_code=401, detail="Token missing")
        decoded = jwt.decode(token, key="secret", algorithms="HS256")
        print(f"decoded: {decoded}")
        
        print(f"token found")
        return func(*args, **kwargs)
    return wrapper

def encrypt_password(password:str):
    print(f"password is being encrypted...")
    password_bytes = password.encode("utf-8")
    hashed_pw = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    print(f"password encrypted successfully")
    return hashed_pw