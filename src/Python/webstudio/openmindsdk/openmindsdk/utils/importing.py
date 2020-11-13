import os, sys, shutil, json, yaml
import subprocess
import random
import string
from urllib.request import Request, urlopen
import re


def sub_unvalid_chars(s: str):
    "replace the unvalid chars by underline"
    return re.sub(r'\W+', '_', s)


def f(string):
    "like the f-string in 3.6+"
    frame = sys._getframe(1)
    return string.format(**frame.f_locals)


def notify(content: dict, url):
    params = json.dumps({'data': content}).encode('utf8')
    print('backend says:', params)
    req = Request(url, data=params, headers={'content-type': 'application/json'})
    try:
        urlopen(req)
    except Exception as e:
        print(e)


def pretty_print(content, format_: str='yaml', fp=sys.stdout, blank_line=True):
    assert format_ in ['json', 'yaml']
    if format_ == 'yaml':
        yaml.dump(content, fp, default_flow_style=False)
    if format_ == 'json':
        json.dump(content, fp, indent=4, default=str)
    if blank_line:
        fp.write('\n')


def run_command(command, echo_cmd=True):
    if isinstance(command, str):
        command = command.split()
    if echo_cmd:
        print(' '.join(command))
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')


def random_string(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

