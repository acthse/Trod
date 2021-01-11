"""
    helo.err
    ~~~~~~~~
"""


class HeloError(Exception):
    """Exception that is the base class of all other error exceptions"""

    description = 'Helo internal error'

    def __init__(self, msg=""):
        super().__init__(msg or self.description)


class UnconnectedError(HeloError):
    description = "Database is not connected yet"


class DuplicateConnect(HeloError):
    description = "Database already connected"


class NoColumnNameError(HeloError):
    description = "Column name must be specified outside the model"


class UnSupportedError(HeloError):
    """Exception for not supported operation"""


class FieldInitError(HeloError):
    """Exception for field init error"""


class NoPKError(HeloError):
    """Exception for model missing primary key"""


class DuplicatePKError(HeloError):
    """Exceptions of two primary key fields in the same model"""


class NotAllowedOperation(HeloError):
    """Operation not allowed in helo"""


class DangerousOperation(HeloError):
    """Dangerous operation due to wrong programming"""


class ProgrammingWarning(RuntimeWarning):
    """Some warnings about not being appreciated"""
