from flask import Flask
from flask import send_file
from flask import send_from_directory
from flask import request
from flask import make_response
from flask import redirect
from flask import url_for
from flask import jsonify
from flask import render_template
import json
import os
import platform
import string
import subprocess
import socket
import random
import shutil
import time
import flask
import uuid
import adal
import requests
import config
import sqlite3, re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"../openmindsdk"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../NeuronBlocks"))

from openmindsdk.modelbuilder.taskbuilder import getElements, preview_model, getNeuronElements

from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'OpenMind!'

proc = None
token = None
PORT = 5001
AUTHORITY_URL = config.AUTHORITY_HOST_URL + '/' + config.TENANT
REDIRECT_URI = 'https://webstudiodl.corp.microsoft.com:5001/getAToken'
TEMPLATE_AUTHZ_URL = ('https://login.microsoftonline.com/{}/oauth2/authorize?' +
                      'response_type=code&client_id={}&redirect_uri={}&' +
                      'state={}&resource={}')
GUEST = "guest"

@app.route('/')
def index():
    #if 'access_token' not in flask.session:
    #    return flask.redirect(flask.url_for('login'))
    return redirect('/docview/get_started')

@app.route("/login")
def login():
    auth_state = str(uuid.uuid4())
    flask.session['state'] = auth_state
    authorization_url = TEMPLATE_AUTHZ_URL.format(
        config.TENANT,
        config.CLIENT_ID,
        REDIRECT_URI,
        auth_state,
        config.RESOURCE)
    resp = flask.Response(status=307)
    resp.headers['location'] = authorization_url
    return resp

@app.route("/getAToken")
def main_logic():
    code = flask.request.args['code']
    state = flask.request.args['state']
    if state != flask.session['state']:
        raise ValueError("State does not match")
    auth_context = adal.AuthenticationContext(AUTHORITY_URL)
    token_response = auth_context.acquire_token_with_authorization_code(code, REDIRECT_URI, config.RESOURCE,
                                                                        config.CLIENT_ID, config.CLIENT_SECRET)
    # It is recommended to save this to a database when using a production app.
    flask.session['access_token'] = token_response['accessToken']
    endpoint = config.RESOURCE + '/' + config.API_VERSION + '/me/'
    http_headers = {'Authorization': 'Bearer ' + flask.session.get('access_token'),
                    'User-Agent': 'adal-python-sample',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'client-request-id': str(uuid.uuid4())}
    flask.session['user_data'] = requests.get(endpoint, headers=http_headers, stream=False).json()
    return flask.redirect('/')

@app.route('/docview/media/<docname>')
def getDocmedia(docname=None):
    print(docname)
    return send_file("/assets/doc/media/" + docname)

@app.route('/docview/<docname>')
def getDoc(docname=None):
    #if 'access_token' not in flask.session or not flask.session['user_data']:
    #    return flask.redirect(flask.url_for('login'))
    #return render_template("docview.html", docname=docname, display_name=flask.session['user_data']['displayName'])
    return render_template("docview.html", docname=docname, display_name=GUEST)

@app.route('/open_project/<modelname>/<settings>')
def open_project(modelname=None, settings=None):
    #if 'access_token' not in flask.session or not flask.session['user_data']:
    #    return flask.redirect(flask.url_for('login'))
    readonly = False
    if (str(settings).find('readonly') > -1):
        readonly = True
    #return render_template("projectView.html", model_name=modelname, readonly=readonly, display_name=flask.session['user_data']['displayName'])
    return render_template("projectView.html", model_name=modelname, readonly=readonly, display_name=GUEST)

@app.route('/open_neuron_project/<modelname>/<settings>')
def open_neuron_project(modelname=None, settings=None):
    #if 'access_token' not in flask.session or not flask.session['user_data']:
    #    return flask.redirect(flask.url_for('login'))
    readonly = False
    if (str(settings).find('readonly') > -1):
        readonly = True
    #return render_template("neuronBlocksView.html", model_name=modelname, readonly=readonly, display_name=flask.session['user_data']['displayName'])
    return render_template("neuronBlocksView.html", model_name=modelname, readonly=readonly, display_name=GUEST)

@app.route('/project/<type>')
def project(type=None):
    #if 'access_token' not in flask.session or not flask.session['user_data']:
    #    return flask.redirect(flask.url_for('login'))
    #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
    user_name = GUEST
    if (type == 'private'):
        conn = sqlite3.connect('main.db')
        conn.row_factory = dict_factory
        cur = conn.cursor()
        cur.execute("select * from projects where owner='" + user_name + "'")
    else:
        conn = sqlite3.connect('example.db')
        conn.row_factory = dict_factory
        cur = conn.cursor()
        cur.execute("select * from projects")
    projects_db = cur.fetchall()
    conn.close()
    #return render_template("projectTable.html", projects=projects_db, type=type, display_name=flask.session['user_data']['displayName'])
    return render_template("projectTable.html", projects=projects_db, type=type, display_name=GUEST)

@app.route('/create', methods=['GET', 'POST'])
def getPage():
    #if 'access_token' not in flask.session or not flask.session['user_data']:
    #    return flask.redirect(flask.url_for('login'))
    #return render_template("create.html", display_name=flask.session['user_data']['displayName'])
    return render_template("create.html", display_name=GUEST)


@app.route('/downLoadNiiPackage/<model_name>/<directory_name>', methods=['GET'])
def downLoadNiiPackage(model_name=None, directory_name=None):
    # model_name = request.args.get('model_name')
    # directory_name = request.args.get('directory_name')
    # if directory_name == 'private':
    #     user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
    #     model_dir = './json_files/' + directory_name + '/' + user_name + '/' + model_name + '/'
    # else:
    #     model_dir = './json_files/' + directory_name + '/' + model_name + '/'
    # try:
    #     fp = ZipFile(model_name + ".zip", mode='w', compression= ZIP_DEFLATED)
    #     fp.write(model_dir + model_name + '.py', model_name + '/' + model_name + '.py')
    #     fp.write(model_dir + 'config.yml', model_name + '/config.yml')
    #     fp.write(model_dir + 'search_space.json', model_name + '/search_space.json')
    # except Exception as e:
    #     return jsonify(result="Failed", page_code="download package Failed")
    # fp.close()
    # return jsonify(result="Success")
    # file_name = 'config.py'
    # file_path = './'
    # response = make_response(send_from_directory(file_path, file_name, as_attachment=True))
    # response.headers["Content-Disposition"] = "attachment; filename={}".format(file_path.encode().decode('latin-1'))
    # return response
    if directory_name == 'private':
        #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
        user_name = GUEST
        file_path = './json_files/' + directory_name + '/' + user_name + '/' + model_name + '/'
    else:
        file_path = './json_files/' + directory_name + '/' + model_name + '/'
    zip_name = model_name + '.zip'
    fp = ZipFile(file_path + zip_name, mode='w', compression=ZIP_DEFLATED)
    fp.write(file_path + model_name + '.py', model_name + '/' + model_name + '.py')
    fp.write(file_path + 'config.yml', model_name + '/config.yml')
    fp.write(file_path + 'search_space.json', model_name + '/search_space.json')
    fp.close()
    response = make_response(send_from_directory(file_path, zip_name, as_attachment=True))
    response.headers["Content-Disposition"] = "attachment; filename={}".format(
        zip_name.encode().decode('latin-1'))
    return response

@app.route('/jsonSaver')
def jsonSaver():
    model_json = json.loads(request.args.get('json_string'))
    model_name = request.args.get('url').split('/')[-2]
    directory_name = request.args.get('directory_name')
    if directory_name == 'private':
        #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
        user_name = GUEST
        model_dir = './json_files/' + directory_name + '/' + user_name + '/' + model_name + '/'
    else:
        model_dir = './json_files/' + directory_name + '/' + model_name + '/'
    f = open(model_dir + model_name + ".json", "w")
    json.dump(model_json, f)
    f.close()
    return jsonify(result=model_json)

@app.route('/nniConfigSaver')
def nniConfigSaver():
    nni_params = json.loads(request.args.get('nni_params'))
    model_name = request.args.get('url').split('/')[-2]
    directory_name = request.args.get('directory_name')
    if directory_name == 'private':
        #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
        user_name = GUEST
        model_dir = './json_files/' + directory_name + '/' + user_name + '/' + model_name + '/'
    else:
        model_dir = './json_files/' + directory_name + '/' + model_name + '/'
    try:
        f = open(model_dir + "search_space.json", "w")
        json.dump(json.loads(nni_params['searchSpace']), f)
        f = open(model_dir + "config.yml", "w")
        f.write("authorName: " + nni_params["authorName"] + "\n")
        f.write("experimentName: " + nni_params["experimentName"] + "\n")
        f.write("trialConcurrency: " + nni_params["trialConcurrency"] + "\n")
        f.write("maxExecDuration: " + nni_params["maxExecDuration"] + "\n")
        f.write("maxTrialNum: " + nni_params["maxTrialNum"] + "\n")
        f.write("trainingServicePlatform: " + nni_params["trainingServicePlatform"] + "\n")
        f.write("searchSpacePath: " + nni_params["searchSpacePath"] + "\n")
        f.write("useAnnotation: " + nni_params["useAnnotation"] + "\n")
        f.write("tuner: \n")
        f.write("  builtinTunerName: " + nni_params["builtinTunerName"] + "\n")
        f.write("  classArgs: \n")
        f.write("    optimize_mode: " + nni_params["optimizeMode"] + "\n")
        f.write("trial: \n")
        f.write("  command: " + nni_params["command"] + "\n")
        f.write("  codeDir: " + nni_params["codeDir"] + "\n")
        f.write("  gpuNum: " + nni_params["gpuNum"] + "\n")
    except Exception as e:
        return jsonify(result="Failed", page_code="generate nni config Failed")
    f.close()
    return jsonify(result='Success')

@app.route('/jsonPreview', methods = ['GET', 'POST'])
def jsonPreview():
    model_name = request.form.get('model_name')
    directory_name = request.form.get('directory_name')
    json_string = request.form.get('json_string')
    if directory_name == 'private':
        #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
        user_name = GUEST
        file_dir = './json_files/' + directory_name + '/' + user_name + '/' + model_name + '/'
    else:
        file_dir = './json_files/' + directory_name + '/' + model_name + '/'
    model_json = json.loads(request.form.get('json_string'))
    result, name_key_dict, layer_name_dict = preview_model(model_json, file_dir)  ##add new parameter rand_dir
    
    if result == 'Succeed':
        try:
            code_filename = model_name + '.py'
            summary_filename = model_name + '.summary.txt'
            # get code of model
            code_f = open(file_dir + code_filename, "r")
            lines = code_f.readlines()
            code_string = "".join(lines)
            code_f.close()
            # get code of model
            model_shape = ""
            if os.path.exists(file_dir + summary_filename):
                
                # return jsonify(result="Failed", page_code=(file_dir + summary_filename))
                summary = parseSummaryFile(file_dir + summary_filename, name_key_dict, layer_name_dict)
                if len(summary) > 0:
                    model_shape = summary
            return jsonify(result=result, page_code=code_string, model_shape=model_shape)
        except Exception as e:
            return jsonify(result="Failed", page_code="File Error!")
    else:
        return jsonify(result="Failed", page_code=str(result))

@app.route('/jsonLoader')
def jsonLoader():
    directory_name = request.args.get('directory_name')
    model_name = request.args.get('model_name')
    if directory_name == 'private':
        #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
        user_name = GUEST
        model_dir = './json_files/' + directory_name + '/' + user_name + '/' + model_name + '/'
    else:
        model_dir = './json_files/' + directory_name + '/' + model_name + '/'
    file_name = model_dir + model_name + ".json"
    if os.path.exists(file_name):
        f = open(file_name, "r")
        model_json = json.load(f)
        f.close()
        return jsonify(result=model_json)
    else:
        return jsonify(result="")

@app.route('/getNodeData')
def getNodeData():
    # if 'keras_elements' not in flask.session:
    #     flask.session['keras_elements'] = getElements()['Keras']
    # elements = flask.session['keras_elements']
    elements = getElements()['Keras']
    groupTitleArray = list(elements.keys())
    groupTitleArray.sort()
    count = 1
    nodeData = []
    fillArray = ['b787b6', 'c7b8a1', '8696a6', '8073b5', '9ba8b8', '4484a0', 'd22a2a', 'd4bd7a', 'f38d8d',
                 '91bf93', '559a94', '795548', 'e8b418', '732722']
    for groupTitle in groupTitleArray:
        nodeData.append([{'key': count}, {'title': groupTitle},
                         {'pic_source': "/assets/img/goJs/" + fillArray[count % 14] + ".png"}])
        nodeTitleArray = []
        nodeMap = {}
        for nodeInfo in elements[groupTitle]['bricks']:
            nodeTitleArray.append(nodeInfo['title'])
            nodeMap[nodeInfo['title']] = nodeInfo
        nodeTitleArray.sort()
        for nodeTitle in nodeTitleArray:
            nodeData.append(createNode(nodeMap[nodeTitle], count, fillArray[count % 14]))
        count = count + 1
    return jsonify(result=nodeData)

@app.route('/getNeuronBlocksData')
def getNeuronBlocksData():
    # if 'neuron_elements' not in flask.session:
    #     flask.session['neuron_elements'] = getNeuronElements()['NeuronBlocks']
    # elements = flask.session['neuron_elements']
    elements = getNeuronElements()['NeuronBlocks']
    groupTitleArray = list(elements.keys())
    groupTitleArray.sort()
    count = 1
    nodeData = []
    fillArray = ['b787b6', 'c7b8a1', '8696a6', '8073b5', '9ba8b8', '4484a0', 'd22a2a', 'd4bd7a', 'f38d8d',
                 '91bf93', '559a94', '795548', 'e8b418', '732722']
    for groupTitle in groupTitleArray:
        nodeData.append([{'key': count}, {'title': groupTitle},
                         {'pic_source': "/assets/img/goJs/" + fillArray[count % 14] + ".png"}])
        nodeTitleArray = []
        nodeMap = {}
        for nodeInfo in elements[groupTitle]['bricks']:
            nodeTitleArray.append(nodeInfo['title'])
            nodeMap[nodeInfo['title']] = nodeInfo
        nodeTitleArray.sort()
        for nodeTitle in nodeTitleArray:
            nodeData.append(createNode(nodeMap[nodeTitle], count, fillArray[count % 14]))
        count = count + 1
    return jsonify(result=nodeData)

@app.route('/run_jupyter', methods=['GET', 'POST'])
def run_jupyter():
    model_name = request.args.get('model_name')
    directory_name = request.args.get('directory_name')
    if directory_name == 'private':
        #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
        user_name = GUEST
        url = 'json_files/' + directory_name + '/' + user_name + '/' + model_name + '/' + model_name + '.ipynb'
    else:
        url = 'json_files/' + directory_name + '/' + model_name + '/' + model_name + '.ipynb'
    global proc, port, token
    if proc is None or port is None or token is None or (not is_pid_running(proc.pid)):
        try:
            token = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))
            target_path = os.path.join("..", "GoJSDemo")
            port = 5002
            proc = subprocess.Popen(
                ['jupyter', 'notebook', '--allow-root', '--no-browser', '--notebook-dir=%s' % target_path,
                 '--NotebookApp.token=%s' % token, '--port=5002', '--ip=0.0.0.0'])
            time.sleep(5)
        except Exception as e:
            return jsonify(result="Failed", page_code=str(e))
    page_code = {'port': port, 'token': token, 'ip': get_host_ip(), 'url': url}
    return jsonify(result="Succeed", page_code=page_code)

@app.route('/stop_jupyter', methods=['GET', 'POST'])
def stop_jupyter():
    print('test stop jupyter')
    global proc, port, token
    if proc is None or (not isinstance(proc, subprocess.Popen)):
        page_code = "No alive juypter server!"
        return jsonify(result="Failed", page_code=page_code)
    try:
        if platform.system() == 'Windows':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(proc.pid)])
        else:
            subprocess.call(['kill', '-9', str(proc.pid)])
        port = None
        token = None
        proc = None
    except Exception as e:
        return jsonify(result="Failed", page_code=str(e))
    page_code = 'Successfully shut down jupyter!'
    return jsonify(result="Succeed", page_code=page_code)

@app.route('/delete_project')
def delete_project():
    #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
    user_name = GUEST
    model_name = request.args.get('model_name')
    conn = sqlite3.connect('main.db')
    cur = conn.cursor()
    file_dir = './json_files/private/' + user_name + '/' + model_name
    try:
        cur.execute("delete from projects where name='" + model_name + "'")
        shutil.rmtree(file_dir)
    except Exception as e:
        conn.close()
        return jsonify(result="Failed", page_code="delete failed")
    conn.commit()
    conn.close()
    return jsonify(result="Succeed", page_code="delete succeed")

@app.route('/create_project', methods=['GET', 'POST'])
def create_project():
    user_name = GUEST
    #user_name = flask.session['user_data']['userPrincipalName'].replace('@microsoft.com', '')
    name = request.args.get('name')
    desc = request.args.get('desc')
    type = request.args.get('type')
    if name == '':
        return jsonify(result="Failed", page_code="project name Can not be empty!")
    conn = sqlite3.connect('main.db')
    conn.row_factory = dict_factory
    cur = conn.cursor()
    cur.execute("select * from projects where owner='" + user_name + "'")
    project_list = cur.fetchall()
    name_list = []
    for item in project_list:
        name_list.append(item['name'])
    if name in name_list:
        return jsonify(result="Failed", page_code="project name already exists!")

    cur.execute('insert into projects (name,desc,public,time,filename,type,owner) values(?,?,?,?,?,?,?)',
                (name, desc, 'YES', time.strftime('%Y-%m-%d', time.localtime(time.time())), name, type, user_name))
    from shutil import copyfile

    file_dir = './json_files/private/' + user_name + '/'
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_dir = file_dir + name + '/'
    os.mkdir(file_dir)
    copyfile('json_files/' + type + '_empty.json', file_dir + name + '.json')
    conn.commit()
    conn.close()
    return jsonify(result="Succeed")



@app.route('/dist/<path:path>')
def dist_static(path):
    return send_from_directory("dist", path)

@app.route('/assets/<path:path>')
def assets_static(path):
    return send_from_directory("assets", path)

@app.route('/bower_components/<path:path>')
def bower_components_static(path):
    return send_from_directory("bower_components", path)

@app.route('/plugins/<path:path>')
def plugins_static(path):
    return send_from_directory("plugins", path)



def createNode(nodeInfo, parent, fill):
    nodeObject = []
    nodeObject.append({'parent': parent})
    nodeObject.append({'title': nodeInfo['title']})
    nodeObject.append({'type': nodeInfo['title']})
    nodeObject.append({'id': nodeInfo['id']})
    nodeObject.append({'framework': nodeInfo['framework']})
    nodeObject.append({'class_name': nodeInfo['class_name']})
    nodeObject.append({'class': nodeInfo['class']})
    nodeObject.append({'leftArray': []})
    nodeObject.append({'rightArray': []})
    if nodeInfo['title'] == 'KerasModelInput' or nodeInfo['title'] == 'SubGraphIn' or nodeInfo[
        'title'] == 'NeuronInput':
        nodeObject.append({'topArray': []})
        nodeObject.append({'bottomArray': [{"portColor": "#94bdef", "portId": "bottom0"}]})
    elif nodeInfo['title'] == 'KerasModelOutput' or nodeInfo['title'] == 'SubGraphOut' or nodeInfo[
        'title'] == 'NeuronOutput':
        nodeObject.append({'topArray': [{"portColor": "#94bdef", "portId": "top0"}]})
        nodeObject.append({'bottomArray': []})
    else:
        nodeObject.append({'topArray': [{"portColor": "#94bdef", "portId": "top0"}]})
        nodeObject.append({'bottomArray': [{"portColor": "#94bdef", "portId": "bottom0"}]})
    nodeObject.append({'output_shape': ''})
    necessary = []
    for item in nodeInfo['parameters']:
        if item['config']['default'] == 'Empty':
            nodeObject.append({item['id']: ''})
            necessary.append(item['id'])
        else:
            nodeObject.append({item['id']: item['config']['default']})
    nodeObject.append({'fill': "#" + fill})
    nodeObject.append({'necessary': necessary})
    return nodeObject

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def buildRandDir():
    rand_dir = str(time.time()).replace('.', '')
    seeds = "1234567890abcdefghijklmnopqrstuvwxyz"
    rand_val = []
    for i in range(10):
        a = random.choice(seeds)
        rand_val.append(str(a))
    rand_dir += ''.join(rand_val)
    return rand_dir

def parseSummaryFile(summary_filename, name_key_dict, layer_name_dict):
    model_shape = {}  # {key: output_shape}
    with open(summary_filename, "r") as summary_f:
        for line in summary_f.readlines():
            if '(' in line:
                line = line.strip().split('(')
                if len(line) == 3:
                    layer_name = line[0].strip().replace('_', '')
                    shape = line[2].split(')')[0].strip()
                    # bind output-shape to key
                    if layer_name in layer_name_dict.keys():  # for Layer
                        instance_name = layer_name_dict[layer_name]
                        key = name_key_dict[instance_name]
                        model_shape[key] = '(' + shape + ')'
    return model_shape

def check_password(hashed_password, user_password):
    return hashed_password == user_password

def validate(username, password):
    con = sqlite3.connect('main.db')
    completion = False
    with con:
        cur = con.cursor()
        cur.execute("SELECT * FROM Users")
        rows = cur.fetchall()
        for row in rows:
            dbUser = row[0]
            dbPass = row[1]
            if dbUser == username:
                completion = check_password(dbPass, password)
    con.close()
    return completion

def is_pid_running(pid):
    return (_is_pid_running_on_windows(pid) if platform.system() == "Windows"
            else _is_pid_running_on_unix(pid))

def _is_pid_running_on_unix(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def _is_pid_running_on_windows(pid):
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(1, 0, pid)
    if handle == 0:
        return False

    exit_code = ctypes.wintypes.DWORD()
    is_running = (
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0)
    kernel32.CloseHandle(handle)

    return is_running or exit_code.value == _STILL_ACTIVE

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT, ssl_context='adhoc')
