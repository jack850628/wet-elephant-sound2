class JException(Exception):
    def __init__(self, message, code = 490):
        super().__init__(message)
        self.message = message
        self.code = code