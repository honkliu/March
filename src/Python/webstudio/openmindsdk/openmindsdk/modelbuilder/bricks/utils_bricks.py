'''
{
  "SourceName": "Utils for easy use",
  "Url": "NA",
  "Description": "We provide some functions to happy use"
}
'''
import subprocess
import os
import sys
import importlib
import random
import string
import shutil
from os.path import isfile, join, basename, splitext, dirname, exists
import json
import tarfile
import yaml
from typing import TextIO

# from
from ..utils import Brick, NodeBrick, ModelBrick, DataReaderBrick, ModelCoder, deprecated

from ...utils.downloader import HttpDownloader, HdfsDownloader, KaggleDownloader
from ...utils import run_command, safe_copy, listdir2, decompress_file, split_data_folder, sub_unvalid_chars, f, \
    FuncCoder
from ...utils.openpai import OpenPAI


class Starter(Brick):
    """
    starter is the first brick that defines the parameters of the graph
    """
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('input-labels', [])
        self.set('role', 'starter')
        self.set('trial-id', 'trial-0')
        self.set('notify-url', 'http://127.0.0.1:5000/push')
        self.set('project-name', 'My Project')
        self.set('graph-name', 'my graph')
        self.register_fe_param('task-type', 'image-classification')
        self.register_fe_param('storage-root', 'tasks')
        self.register_fe_param('working-directory', 'workings')

    def __customize__(self, config):
        super().__customize__(config)  # update by config
        self.update_trial_id(self.get('trial-id'))

    def update_trial_id(self, trial_id: str):
        """
        for multiple trial settings, each trial will have a unique anme and working directory
        :param trial_id:
        :return:
        """
        self.set('trial-id', trial_id)
        if self.get('graph-name') is not None:
            self.set('trial-storage', os.path.join(self.get('storage-root'), self.get('graph-name'), self.get('trial-id')))
            self.set('trial-working', os.path.join(self.get('working-directory'), self.get('graph-name'), self.get('trial-id')))

    def update_notify_url(self, notify_url):
        """
        update the property
        :param notify_url:
        :return:
        """
        self.set("notify-url", notify_url)

    def get_display_name(self):
        return "ProjectSettings"


class Executor(Brick):
    "the triger of executing"
    FILENAME = __file__  
    
    def __declare__(self):
        super().__declare__()
        self.set('role', 'executor')
        self.set('output-labels', [])
        

class SubGraphIn(ModelBrick):
    "input of a model"
    FILENAME = __file__
    BASEMODEL = None

    def __declare__(self):
        super().__declare__()
        self.set('role', 'model-component-input')
        self.set('input-labels', [])
        self.register_fe_param('x-shape', [None])
        self.register_fe_param('shape-include-batch-size', True)
        self.register_fe_param('variable-name', 'X')
        self.register_fe_param('order', 0)

    def get_x_shape(self, include_batch: bool=True):
        """
        :param include_batch:
        :return: shape of x, including batch-size if include_batch is true
        """
        batch_shape = self.get('x-shape')
        if not self.get('shape-include-batch-size'):
            batch_shape = [None] + list(batch_shape)
        return tuple(batch_shape if include_batch else batch_shape[1:])

    def port_name(self, *args):
        """
        if {variable-name} is defined, it will be used to declare
        :param name_:
        :return:
        """
        return sub_unvalid_chars(self.get('variable-name'))

    def generate_code_inner(self, gen: ModelCoder):
        if self.BASEMODEL is not None:
            gen.add_module(self.BASEMODEL)
        x = self.get('output-labels')
        assert len(x) == 1, 'only support single input'
        x = self.port_name()
        gen.inputs.update({x: self.get('instance-name')})
        gen.add_func_args(x, self.get('order'))
        return x

    def declare_variable(self, gen:ModelCoder):
        raise NotImplementedError


class SubGraphOut(ModelBrick):
    "output of a model"
    FILENAME = __file__
    BASEMODEL = None

    def __declare__(self):
        super().__declare__()
        self.set('role', 'model-component-output')
        self.set('output-labels', [])
        self.register_fe_param('model-id', 'model-0')
        self.register_fe_param('variable-name', 'Y')
        self.set('is-subgraph-output', True)

    def generate_code(self):
        code_gen = ModelCoder('get_' + self.id(), FuncCoder)
        self.traceback_run(state=dict(), code_gen=code_gen)
        return code_gen

    def port_name(self, name_: str):
        """
        if {variable-name} is defined, it will be used to declare 
        :param name_:
        :return:
        """
        var_names = self.get('variable-name')
        var_names = [var_names] if isinstance(var_names, str) else var_names
        var_dict = {k:v for k,v in zip(self.get('input-labels'), var_names)}
        return sub_unvalid_chars(var_dict[name_])

    def generate_code_inner(self, gen: ModelCoder):
        if self.BASEMODEL is not None:
            gen.add_module(self.BASEMODEL)
        in_ = self.get_inputs(flatten=False)
        for port, connect in in_.items():
            y_ = self.port_name(port)
            gen.outputs.append(y_)
            gen.add_a_line(f('{y_} = {connect}'))
        out_ = ', '.join(gen.outputs)
        gen.add_a_line(f('return {out_}'))

    def generate_model_wrapper(self, gen: ModelCoder):
        return None

    def generate_model_summary(self, gen: ModelCoder):
        return None

    def generate_model_callback(self, gen: ModelCoder):
        return None

    def generate_default_params(self, gen: ModelCoder):
        return None

    def generate_get_data(self, model_parameters):
        return None

    def id(self):
        return sub_unvalid_chars(self.get('model-id'))


class SubGraphSiSo(ModelBrick):
    "single input single output subgraph"
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.register_fe_param('subgraph-id', '')
        self.register_fe_param('graph-name', None)
        self.register_fe_param('project-name', None)

    def generate_code_inner(self, gen: ModelCoder):
        func_name = 'get_{}'.format(sub_unvalid_chars(self.get('subgraph-id')))
        module_ = self.get('graph-name')
        if module_ is not None:
            module_ = sub_unvalid_chars(module_)
        if module_ is None:
            filename = None
        elif self.get('project-name') is None:
            filename = os.path.join('..', self.get('graph-name'), module_ + '.py')
        else:
            filename = os.path.join('..', '..', self.get('project-name'), self.get('graph-name'), module_+'.py')
        if filename is not None:
            safe_copy(os.path.join(self.parent.stages[0].get('trial-storage'), filename), self.parent.node.register(os.path.basename(filename)))
            func_name = '{}.{}'.format(module_, func_name)
            gen.add_module(module_)
        else:
            gen.add_dependency(func_name)
        in_ = self.get_inputs()
        if '[{' in in_:
            in_ = in_[2:-2]
        if '[' in in_:
            in_ = in_[1:-1]
        out_ = self.variable(self.get('output-labels')[0])
        gen.add_a_line(f('{out_} = {func_name} ({in_})'))


@deprecated
class DataReaderPlaceHolder(DataReaderBrick):
    FILENAME = __file__

@deprecated
class DataSplitter(Brick):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('role', 'downloader') 
        self.set('source-directory', 'origin_data')
        self.set('destination-directories', ['train', 'val'])
        self.set('split-ratios', [0.9, 0.1])
        self.set('overwrite-when-dest-exists', True)
        
        self.add_fe_parameters('source-directory', 'destination-directories', 'split-ratios', 'overwrite-when-dest-exists')
        
    def traceback_run(self, state):
        super().traceback_run(state)
        relative_path = {
            'src': self.parent.node.local(self.get('source-directory')),
            'dest': [self.parent.node.local(d) for d in self.get('destination-directories')]
        }
        split_data_folder(relative_path['src'], relative_path['dest'], self.get('split-ratios'))

@deprecated
class DownloaderBrick(Brick):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('role', 'downloader')
        self.batch_set(['download-cache-directory'], ['./data'])
        self.set('unarchive-instruction', {})
        self.set('sources', [{"source_url": None, "download_filename": None, "destination": None}])
        self.set('hdfs_sources', [{"source_url": None, "user": None, "root": None, "download_filename": None, "destination": None}])
        self.set('kaggle_sources', [{'download-cmd': None, 'kaggle-bin-path': None, 'download-filename': None, "destination": None, 'kaggle-config': None}])
        #self.set('sources', [])
        #self.set('hdfs_sources', [])
        #self.set('kaggle_sources', [])
        self.set('call_back_func', '')
        self.add_fe_parameters("sources", type_="list")
        self.add_fe_parameters("hdfs_sources", type_="list")
        self.add_fe_parameters("kaggle_sources", type_="list")

        self.http_downloader = HttpDownloader()
        self.hdfs_downloader = HdfsDownloader()
        self.kaggle_downloader = KaggleDownloader()

    def __customize__(self, config: dict):
        super().__customize__(config)
        self.unarchive_cmd = {
            'tgz': 'tar xvfz',
            'tar': 'tar xvf',
            'zip': 'unzip -n',
            'gzip': 'gunzip', 'gz': 'gunzip'
        }
        self.unarchive_cmd.update(self.get('unarchive-instruction'))

    def traceback_run(self, state):
        super().traceback_run(state)
        
        # download data according to its protocol
        http_cfgs = self.get('sources')
        if len(http_cfgs) > 0:
            self.download(http_cfgs, self.http_downloader)

        hdfs_cfgs = self.get('hdfs_sources')
        if len(hdfs_cfgs) > 0 and hdfs_cfgs[0]['source_url']:
            self.download(hdfs_cfgs, self.hdfs_downloader)

        kaggle_cfgs = self.get('kaggle_sources')
        if len(kaggle_cfgs) > 0 and kaggle_cfgs[0]['download-cmd']:
            self.download(kaggle_cfgs, self.kaggle_downloader)

    def download(self, cfgs, downloader):
        for cfg in cfgs:
            cfg = downloader.init(cfg)
            downloaded_file_or_dir = self.download_each(cfg, downloader)
            if downloaded_file_or_dir:
                dest_file_or_dir = self.copy_to_working_directory(downloaded_file_or_dir)
            dest_path = os.path.join(self.parent.node.local(), dest_file_or_dir)
            if isfile(dest_path):
                self.unarchive(cfg, self.parent.node.local(), dest_file_or_dir)
            else:
                #unarchive files in dir
                for r, _, files in os.walk(dest_path):
                    for f in files:
                        self.unarchive(cfg, dest_path, f)

        if 'call_back_func' in cfg and cfg['call_back_func']:
            self.run_callback(cfg['call_back_func'])

    def download_each(self, cfg, downloader):
        currdir = os.getcwd()
        try:
            cache_dir = self.get('download-cache-directory')
            if 'download_filename' in cfg:
                dest_file_or_dir = cfg['download_filename']
                print('test existence of %s' % (dest_file_or_dir,))
            else:
                dest_file_or_dir = ''
            if not isfile(join(cache_dir, dest_file_or_dir)):
                os.makedirs(cache_dir, exist_ok=True)
                os.chdir(cache_dir)
                dest_file_or_dir = downloader.retrive_file(cfg)
        except Exception as identifier:
            print(identifier)
        finally:
            os.chdir(currdir)
            return os.path.join(cache_dir, dest_file_or_dir)

    def copy_to_working_directory(self, src_file_or_dir):
        dest_path = self.parent.node.local(basename(src_file_or_dir))
        if not exists(dest_path):
            if isfile(src_file_or_dir):
                safe_copy(src_file_or_dir, dest_path)
            else:
                shutil.copytree(src_file_or_dir, dest_path)

        return basename(dest_path)

    def unarchive(self, cfg: dict, dest_dir: str, filename: str):
        workdir = dest_dir
        print('unarchiving %s in %s' % (filename, workdir))
        currdir = os.getcwd()

        dest = cfg['destination'] if 'destination' in cfg else ''
        try:
            os.chdir(workdir)
            if dest is not None and os.path.exists(dest):
                return join(workdir, dest)
            assert isfile(filename), 'cannot find %s in %s' % (filename, workdir)
            if tarfile.is_tarfile(filename):
                with tarfile.open(filename, "r") as tar:
                    tar.extractall()
            else:
                name, ext = splitext(filename)
                while len(ext) > 0 and ext[1:] in self.unarchive_cmd:
                    for out in run_command(self.unarchive_cmd[ext[1:]].split() + [filename]):
                        print (out.rstrip().decode('utf-8'))
                    filename = name
                    name, ext = splitext(filename)
                if dest is None:
                    dest = name
            if dest is None:
                name, ext = splitext(filename)
                while len(ext) > 0 and ext[1:] in self.unarchive_cmd:
                    filename = name
                    name, ext = splitext(filename)
                dest = name
        except Exception as identifier:
            print(identifier)
        finally:
            os.chdir(currdir)
        return join(workdir, dest)

    def run_callback(self, cmd):
        exec(cmd)


@deprecated
class KaggleBrick(Brick):
    '''
    Kaggle module to download data
    https://github.com/Kaggle/kaggle-api
    '''
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('role', 'downloader')
        self.set('install-instruction', 'pip install --user kaggle')
        self.batch_set(['download-cache-directory'], ['./data'])

        self.set('kaggle-bin-path', '')
        self.set('download-cmd', None)
        self.set('download-filename', None)
        self.set('kaggle-config', None)
        self.add_fe_parameters('download-cmd', 'kaggle-bin-path', 'download-filename', 'kaggle-config')

    def download(self):
        currdir = os.getcwd()
        try:
            #install kaggle module
            if not importlib.util.find_spec('kaggle'):
                cmd = self.get('install-instruction')
                #out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
                #print (out.rstrip().decode('utf-8'))
                for out_ in run_command(cmd):
                    print (out_.rstrip().decode('utf-8'))

            if not self.get('kaggle-bin-path'):
                self.set('kaggle-bin-path', self.get_default_kaggle_bin_path())

            self.set_kaggle_config()

            download_dir = os.path.join(currdir, self.parent.node.local())
            filename = self.get('download-filename')
            if not filename or not isfile(join(download_dir, filename)):
                os.makedirs(download_dir, exist_ok=True)
                os.chdir(download_dir)

                cmd = self.get_cmd(download_dir, filename)
                print('run kaggle command: {}'.format(cmd))
                for out_ in run_command(cmd):
                    print (out_.rstrip().decode('utf-8'))

                os.chdir(currdir)
                self.unarchive(download_dir, filename)
        except Exception as identifier:
            print(identifier)
        finally:
            os.chdir(currdir)
            return download_dir

    def set_kaggle_config(self):
        default_path = '~/.kaggle/kaggle.json'
        if not isfile(default_path):
            config = self.get('kaggle-config')
            if config:
                with open(default_path, 'w') as f:
                    f.write(config)


    def get_cmd(self, workdir, filename):
        cmd = os.path.join(self.get('kaggle-bin-path'), self.get('download-cmd'))
        if workdir:
            cmd = "{} -p {}".format(cmd, workdir)
        if filename:
            cmd = "{} -f {}".format(cmd, filename)
        return cmd

    def get_default_kaggle_bin_path(self):
        default_path = '~/.local/bin'
        if sys.platform == 'win32' or sys.platform == 'win64':
            kaggle_module = importlib.import_module('kaggle')
            dir_name = dirname(dirname(dirname(kaggle_module.__file__)))
            default_path = os.path.join(dir_name, 'Scripts')

        return default_path

    def unarchive(self, download_dir: str, filename: str):
        if filename:
            src_files = [filename]
        else:
            src_files = os.listdir(download_dir)

        try:
            currdir = os.getcwd()
            os.chdir(download_dir)
            for filename in src_files:
                print('unarchiving %s in %s' % (filename, workdir))
                decompress_file(filename)
        except Exception as identifier:
            print(identifier)
        finally:
            os.chdir(currdir)


class ComputeNodeLocal(NodeBrick):
    """ run and storage the jobs in local environment"""
    FILENAME = __file__

    def copy_result_files(self):
        if self.pget('trial-storage') == self.pget('trial-working'):
            return
        for f in self.files.keys():
            if self.copied[f]:
                continue
            if os.path.exists(self.local(f)):
                safe_copy(self.local(f), self.storage(f))
                self.copied[f] = True
            else:
                print('{} doesnot exist'.format(f))

    def submit_job(self, commands: list=[]):
        import threading
        def run(commands):
            for out_ in run_command(commands):
                print (out_.rstrip().decode('utf-8'))
        threading.Thread(target=run, args=(commands,)).start()
        return ''

    def register_all(self):
        for f in listdir2(self.storage(), full_path_out=False):
            self.register(f)


class ComputeNodePAI(NodeBrick):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set("pai-config", {'rest_server_socket': '', 'hdfs_web_socket': '', 'web_portal': '', 'user': '', 'password': ''})
        self.add_fe_parameters("pai-config::*")

    def get_storage(self, f):
        storage_path = os.path.join("/user/dragdrop/", self.get('trial-storage'), f)
        return storage_path.replace('\\', '/')

    def get_code_dir(self):
        code_dir = os.path.join("$PAI_DEFAULT_FS_URI/user/dragdrop/", self.get('trial-storage'))
        return code_dir.replace('\\', '/')

    def get_job_name(self):
        return "dragdrop_" + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))

    def get_hdfs_url(self):
        # http://10.151.40.179:50070/explorer.html#/user/dragdrop/tasks/flower_pai/trial-0
        return self.get("pai-config")["hdfs_web_socket"] + "/explorer.html#/user/dragdrop/" + self.get('trial-storage')

    def copy_result_files(self):
        pai_config = self.get("pai-config")
        self.openpai = OpenPAI(config=pai_config)
        for f in self.result_files:
            self.openpai.upload(self.local(f), self.get_storage(f))

    def submit_job(self, commands: list=[]):
        # call rest api to submit to pai
        job_config = {}
        job_config["jobName"] = self.get_job_name()
        job_config["image"] = "tagineerdai/cuda90-deep-learning:latest"
        job_config["codeDir"] = self.get_code_dir()

        task_roles = []
        task_role = {}
        task_role["name"] = job_config["jobName"] + "_train"
        task_role["taskNumber"] = 1
        task_role["cpuNumber"] = 8
        task_role["memoryMB"] = 32768
        task_role["gpuNumber"] = 1

        # replace config
        index = commands.index("--config")
        commands[0] = 'python3'
        assert index >= 0
        commands[index + 1] = '/root/Project/' + self.get('trial-id') + "/" + os.path.basename(commands[index + 1])
        task_role["command"] = "cd /root && pip3 install hdfs psutil && hadoop fs -get $PAI_DEFAULT_FS_URI/user/dragdrop/code/ModelBuilder.BE.zip && unzip ModelBuilder.BE.zip && cd /root/ModelBuilder.BE && " + " ".join(commands)
        # task_role["command"] = "jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --NotebookApp.token=abcd"
        task_roles.append(task_role)
        job_config["taskRoles"] = task_roles
        print(json.dumps(job_config))
        self.openpai.submit_job(job_config)
        pai_config = self.get("pai-config")
        job_url = pai_config['web_portal'] + "/view.html?jobName=" + job_config["jobName"]
        return job_url

    def register_all(self):
        # fix me
        pass
