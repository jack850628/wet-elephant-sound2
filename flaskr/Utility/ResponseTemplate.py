import json
from flask import Response


def success(message: str = 'success') -> Response:
    return Response(
        json.dumps({
            'result': 'success',
            'message': message
        })
    )

def fail(message: str) -> Response:
    return Response(
        json.dumps({
            'result': 'fail',
            'error_code': message
        })
    )