import os
import sys
import urllib
import importlib
try:
    from hdfs import InsecureClient
except:
    print('no hdfs')
from os.path import isfile, join, basename, splitext, dirname, exists

class HttpDownloader():
    FILENAME = __file__

    def init(self, cfg):
        source_url = cfg['source_url']
        if not cfg['download_filename']:
            cfg['download_filename'] = basename(urllib.parse.urlparse(source_url).path)
        return cfg

    def retrive_file(self, cfg):
        fname = cfg['download_filename']
        url = cfg['source_url']
        print("download {} from {}".format(fname, url))
        urllib.request.urlretrieve(url, fname)
        return fname


class HdfsDownloader():
    FILENAME = __file__

    def init(self, cfg):
        url = cfg['source_url']
        user = cfg['user'] if 'user' in cfg else None
        root_path = cfg['root'] if 'root' in cfg else '/'
        if 'download_filename' not in cfg:
            cfg['download_filename'] = basename(url)

        http_protocal_prefix = 'http://'
        if url.startswith(http_protocal_prefix):
            index = url[len(http_protocal_prefix):].find('/') + len(http_protocal_prefix)
            host_port = url[:index]
            cfg['file_path'] = url[index:]
            self.hdfs_client = InsecureClient(url = host_port, user=user, root=root_path)
        return cfg

    def retrive_file(self, cfg):
        url = cfg['source_url']
        print("download hdfs file: {}".format(url))

        if url.startswith('http://'):
            path = cfg['file_path']
            self.hdfs_client.download(path, cfg['download_filename'], overwrite=True)
        elif url.startswith('hdfs://'):
            cmd = 'hadoop fs -get {}'.format(url)
            for out_ in run_command(cmd):
                print (out_.rstrip().decode('utf-8'))
        else:
            raise Exception('Not supported protocal. Only support "http://" or "hdfs://"')

        return basename(url)


class KaggleDownloader():
    FILENAME = __file__

    def init(self, cfg):
        #install kaggle module
        if not importlib.util.find_spec('kaggle'):
            cmd = 'pip install --user kaggle'
            for out_ in run_command(cmd):
                print (out_.rstrip().decode('utf-8'))

        if not 'kaggle-bin-path' in cfg or not cfg['kaggle-bin-path']:
            cfg['kaggle-bin-path'] = self.get_default_kaggle_bin_path()

        self.set_kaggle_config(cfg)
        return cfg

    def set_kaggle_config(self, cfg):
        default_path = '~/.kaggle/kaggle.json'
        if not isfile(default_path) and 'kaggle-config' in cfg:
            config = cfg['kaggle-config']
            if config:
                with open(default_path, 'w') as f:
                    f.write(config)


    def get_cmd(self, cfg):
        cmd = cfg['download-cmd']
        if 'competitions download -c' in cmd:
            dpath = cmd[cmd.find(' -c ') + len(' -c ') : ].replace('/', '-')
        elif 'datasets download -d' in cmd:
            dpath = cmd[cmd.find(' -d ') + len(' -d ') : ].replace('/', '-')
        else:
            raise Exception('Not supported command. Only support "competitions downlaod -c" or "datasets download -d"')

        cmd = os.path.join(cfg['kaggle-bin-path'], cfg['download-cmd'])
        return cmd, dpath

    def get_default_kaggle_bin_path(self):
        default_path = '~/.local/bin'
        if sys.platform == 'win32' or sys.platform == 'win64':
            kaggle_module = importlib.import_module('kaggle')
            dir_name = dirname(dirname(dirname(kaggle_module.__file__)))
            default_path = os.path.join(dir_name, 'Scripts')

        return default_path

    def retrive_file(self, cfg):
        cmd, dpath = self.get_cmd(cfg)

        currdir = os.getcwd()
        try:
            os.makedirs(dpath, exist_ok=True)
            os.chdir(dpath)
            print('run kaggle command: {}'.format(cmd))
            for out_ in run_command(cmd):
                print (out_.rstrip().decode('utf-8'))
        finally:
            os.chdir(currdir)

        return dpath
