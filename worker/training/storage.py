"""Object-store transport and artifact identity shared by all trainers."""
import hashlib
from urllib.parse import quote

import requests


class ObjectStore:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')

    def url(self, action, path):
        path = path.removeprefix('s3://')
        bucket, separator, name = path.partition('/')
        if not separator or not name or bucket not in ('datasets', 'models', 'artifacts'):
            raise ValueError('Expected datasets/<object>, models/<object>, or artifacts/<object>')
        return f'{self.base_url}/api/v1/{action}/{quote(bucket)}/{quote(name, safe="/")}'

    def get(self, path):
        response = requests.get(self.url('download', path), timeout=(5, 60))
        response.raise_for_status()
        return response.content

    def put(self, path, content):
        response = requests.post(
            self.url('upload', path),
            files={'file': ('artifact', content, 'application/octet-stream')},
            timeout=(5, 60),
        )
        response.raise_for_status()


def digest(content):
    return hashlib.sha256(content).hexdigest()


