from cryptography.fernet import Fernet
import os
from common.config import SECRET_KEY_FILE

KEY_FILE = SECRET_KEY_FILE

def load_key():
    """
    Loads the key from the current directory named `secret.key`
    """
    if not os.path.exists(KEY_FILE):
        generate_key()
    return open(KEY_FILE, "rb").read()

def generate_key():
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)

def encrypt_file(file_name):
    """
    Encrypts a file
    """
    key = load_key()
    f = Fernet(key)
    
    with open(file_name, "rb") as file:
        file_data = file.read()
        
    encrypted_data = f.encrypt(file_data)
    
    with open(file_name, "wb") as file:
        file.write(encrypted_data)

def decrypt_file(file_name):
    """
    Decrypts a file and returns the data (bytes)
    """
    key = load_key()
    f = Fernet(key)
    
    with open(file_name, "rb") as file:
        encrypted_data = file.read()
        
    decrypted_data = f.decrypt(encrypted_data)
    return decrypted_data
