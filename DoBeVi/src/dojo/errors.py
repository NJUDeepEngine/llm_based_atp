class BaseError(Exception):
    PREFIX = "Error"

    def __init__(self, message):
        message = f"{self.PREFIX}: {message}"
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message

class RepoInitError(BaseError):
    PREFIX = "RepoInitError"

class DojoInitError(BaseError):
    PREFIX = "DojoInitError"

class DojoCrashError(BaseError):
    PREFIX = "DojoCrashError"

class DojoTacticTimeoutError(BaseError):
    PREFIX = "DojoTacticTimeoutError"