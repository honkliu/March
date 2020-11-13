import os
import errno
import shutil
import fnmatch
import random
import numpy as np
import tarfile, gzip, zipfile
from contextlib import contextmanager
import json
import yaml


def listdir2(folder, c_type='files', full_path_out=True, filter_pat=None):
    '''type can only be root, dirs and files'''
    allowed = ['root', 'dirs', 'files']
    k = allowed.index(c_type)
    out = next(os.walk(folder))[k]
    if filter_pat is not None:
        out = fnmatch.filter(out, filter_pat)
    if full_path_out:
        out = [os.path.join(folder, f) for f in out]
    return out


def files2list(folder:str, files:list=[], prefix: str=''):
    '''return a list of files in the folder (relative paths)'''
    currfiles = listdir2(folder, full_path_out=False)
    nextfolders = listdir2(folder, c_type='dirs', full_path_out=False)
    files.extend([os.path.join(prefix, f) for f in currfiles])
    for nextfolder in nextfolders:
        files2list(os.path.join(folder, nextfolder), files, os.path.join(prefix, nextfolder))
    return files


def file_func(kwargs: dict, func=shutil.copy2, tester: str='dst'):
    try:
        return func(**kwargs)
    except IOError as identifier:
        # ENOENT(2): file does not exist, raised also on missing dest parent dir
        if identifier.errno != errno.ENOENT:
            print(identifier.__dict__)
        assert(tester in kwargs.keys())
        os.makedirs(os.path.dirname(kwargs[tester]), exist_ok=True)
        return func(**kwargs)
    except Exception as identifier:
        print(identifier)
        return None


@contextmanager
def safe_open(filename: str, mode: str='r', **kwargs):
    "if directory of filename doesnot exist, create it first"
    args = dict(kwargs)
    args.update({'file':filename, 'mode':mode})
    fn = file_func(args, func=open, tester='file')
    yield fn
    fn.close()


@contextmanager
def safe_chdir(pth:str):
    "safely change directory to pth, and then go back"
    currdir = os.getcwd()
    try:
        os.chdir(pth)
        yield pth
    finally:
        os.chdir(currdir)


def safe_copy(src: str, dst: str):
    "if directory of filename doesnot exist, create it first"
    return file_func({'src':src, 'dst':dst})


def to_file(content, filename: str):
    with safe_open(filename, 'w') as fp:
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            yaml.dump(content, fp, default_flow_style=False)
        elif filename.endswith('.json'):
            json.dump(content, fp, indent=4)
        else:
            raise ValueError('filename {} is not a recognized format'.format(filename))


def from_file(filename: str):
    with safe_open(filename, 'r') as fp:
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            content = yaml.load(fp)
        elif filename.endswith('.json'):
            content = json.load(fp)
        else:
            raise ValueError('filename {} is not a recognized format'.format(filename))
    return content


def split_data_folder(src_folder:str, dest_folders:list=['train','test'], ratio:list=[], func=shutil.copy2):
    '''split source directory {src_folder} to dest_folders[0], dest_folders[1], ...'''
    try:
        assert np.sum(ratio) == 1.0
        assert len(ratio) == len(dest_folders)
        for dest in dest_folders:
            shutil.rmtree(dest, ignore_errors=True)
        files = files2list(src_folder, list())
        print('%d files loaded from %s' % (len(files), os.path.abspath(src_folder)))
        random.shuffle(files)
        cum = np.round(np.cumsum(ratio) * len(files))
        dest, k = dest_folders[0], 0
        for i, f in enumerate(files):
            file_func({'src':os.path.join(src_folder, f), 'dst':os.path.join(dest, f)}, func)
            if i+1 == cum[k]:
                print('%d files reserved for %s' %(i+1 if k==0 else i+1 - cum[k-1], dest))
                if i==len(files)-1:
                    break
                dest, k = dest_folders[k+1], k+1
    except Exception as identifier:
        print(identifier)
        raise


def train_validation_split_folder(folder: str, validation_split: float):
    dests = [folder+'_'+postfix for postfix in ['train', 'validation']]
    ratio = [1 - validation_split, validation_split]
    return split_data_folder(folder, dests, ratio, func=shutil.copy2)


def decompress_file(filename: str):
    if filename.endswith("tar.gz") or filename.endswith("tgz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
    elif filename.endswith("tar"):
        tar = tarfile.open(filename, "r:")
        tar.extractall()
        tar.close()
    elif filename.endswith("zip"):
        f = open(filename, 'rb')
        zf = zipfile.ZipFile(f)
        zf.extractall()
        f.close()
    elif filename.endswith("gzip") or filename.endswith("gz"):
        read_f = gzip.GzipFile(filename, 'rb')
        content = read_f.read()
        fname, ext = os.path.splitext(filename)
        read_f.close()
        out_f = open(fname, 'wb')
        out_f.write(content)
        out_f.close()
