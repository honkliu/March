"""
{
    "SourceName": "client for OpenPAI",
    "URL": "https://github.com/Microsoft/pai"
}
"""

import json
from requests import request
from urllib import parse
from hdfs import Client

class OpenPAI:
    "client for OpenPAI"

    def __init__(self, config: dict=None, file: str='openpai.json'):
        """config should contain
            - rest_server_socket
            - hdfs_web_socket
            - user
            - password
        """
        if config is None:
            with open(file) as fn:
                config = json.load(fn)
        for key in ['rest_server_socket', 'hdfs_web_socket', 'user', 'password']:
            assert key in config, '%s is not defined for OpenPAI' % (key)
        for key in ['rest_server_socket', 'hdfs_web_socket']:
            assert config[key].startswith('http://'), '%s should have http prefix' % (key)

        self.rest_server_socket = config['rest_server_socket']
        self.hdfs_client = Client(config['hdfs_web_socket'])
        self.config = config

    def __get_token(self, user: str, password: str):
        try:
            response = request(
                    'POST',
                    parse.urljoin(self.rest_server_socket, 'token'),
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    data='username={}&password={}&expiration=30000'.format(user, password),
                )
            if response.status_code == 200:
                return response.json()['token']
            raise Exception(response.reason)
        except Exception as identifier:
            raise Exception(identifier)

    def upload(self, local_path, hdfs_dir, overwrite = True):
        try:
            self.hdfs_client.upload(hdfs_dir, local_path, overwrite=overwrite)
            return True
        except Exception as e:
            print(e)
            return False

    def mkdir(self, hdfs_dir):
        try:
            self.hdfs_client.makedirs(hdfs_dir)
            return True
        except Exception as e:
            return False

    def submit_job(self, job_config):
        self.token = self.__get_token(self.config['user'], self.config['password'])
        response = request('POST',
                           parse.urljoin(self.rest_server_socket, 'jobs'),
                           headers={
                               'Authorization': 'Bearer ' + self.token,
                               'Content-Type': 'application/json'
                           },
                           json=job_config)
        if response.status_code != 200 and response.status_code != 202:
            raise Exception(response.reason)