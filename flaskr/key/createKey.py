import base64, hashlib, random

def createKey():
    return  base64.b64encode(hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).digest()).decode('utf-8')

if __name__ == "__main__":
    print(createKey())