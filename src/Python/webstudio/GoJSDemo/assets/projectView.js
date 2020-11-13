//control model page isnt fisrt open
first_open = true
var nodeDataArray = [];
var modelData = {};
var myDiagram = null;
var hyperparameterElements = ['batch_size', 'epochs', 'loss', 'optimizer', 'dataset', 'validation_split', 'shuffle'];
var optimizerParamsElements = ['lr', 'momentum', 'beta_1', 'beta_2', 'rho', 'decay', 'schedule_decay'];
var nniConfigParamsElements = ['authorName', 'experimentName', 'trialConcurrency', 'maxExecDuration', 'maxTrialNum',
    'trainingServicePlatform', 'searchSpacePath', 'useAnnotation', 'builtinTunerName',
    'optimizeMode', 'command', 'codeDir', 'gpuNum', 'searchSpace'];

function initTable() {
    dataButton.onclick = openData;
    modelButton.onclick = openModel;
    hyperparameterButton.onclick = openHyperparameter;
    openData();
}

//convert diffrent step
function openData() {
    dataButton.style.background = 'lightblue';
    hyperparameterButton.style.background = '';
    modelButton.style.background = '';
    hyperparameterContent.style.display = 'none';
    mainContent.style.display = 'none';
    dataContent.style.display = 'block';
};
function openModel() {
    dataButton.style.background = '';
    modelButton.style.background = 'lightblue';
    hyperparameterButton.style.background = '';
    hyperparameterContent.style.display = 'none';
    dataContent.style.display = 'none';
    if(first_open){
        init();
        first_open = false
    }
    mainContent.style.display = 'block';
};
function openHyperparameter() {
    dataButton.style.background = '';
    hyperparameterButton.style.background = 'lightblue';
    modelButton.style.background = '';
    mainContent.style.display = 'none';
    dataContent.style.display = 'none';
    hyperparameterContent.style.display = 'block';
};

function enableNii(){
    document.getElementById('default-hyperparameter').style.marginLeft="15%";
    document.getElementById('nni-model').style.display="block";
}

function disableNii(){
    document.getElementById('nni-model').style.display="none";
    document.getElementById('default-hyperparameter').style.marginLeft="35%";
}

//order by with same elements
function setNumber() {
    var allNodeArray = myDiagram.model.nodeDataArray;
    var length = allNodeArray.length;
    var newNode = allNodeArray[length - 1];
    var title = newNode.type;
    var sameTitleArray = [];
    for (var index = 0; index < length - 1; index++) {
        if (allNodeArray[index].type === newNode.type) {
            sameTitleArray.push(allNodeArray[index]['title'])
        }
    }
    for (var i = 1; i < 100; i++) {
        var tempTitle = newNode.type + '_' + i;
        if (sameTitleArray.indexOf(tempTitle) >= 0) {
            continue;
        } else {
            title = tempTitle;
            break;
        }
    }
    newNode.title = title;
    myDiagram.model.updateTargetBindings(newNode)
}


//Trigger response of select box
 var optimizerParams = {
    'SGD': ['lr', 'momentum', 'decay'],
    'RMSprop': ['lr', 'rho', 'epsilon', 'decay'],
    'Adagrad': ['lr', 'epsilon', 'decay'],
    'Adadelta': ['lr', 'rho', 'epsilon', 'decay'],
    'Adam': ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay'],
    'Adamax': ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay'],
    'Nadam': ['lr', 'beta_1', 'beta_2', 'epsilon', 'schedule_decay']
};
var datasetParams = {
     'mnist' : 'shape: (None, 28, 28)',
     'fashion_mnist' : 'shape: (None, 28, 28)',
     'cifar10' : 'shape: (None, 32, 32, 3)',
     'cifar100' : 'shape: (None, 32, 32, 3)',
     'imdb' : 'shape: (None, )',
     'reuters' : 'shape: (None, )',
     'boston_housing' : 'shape: (None, 13)',
     'custom' : 'shape: (None, )'
}
function datasetChange() {
    var datasetSelect = document.getElementById("dataset");
    var datasetType = datasetSelect.options[datasetSelect.selectedIndex].value;
    document.getElementById('shapeDesc').innerText = datasetParams[datasetType];
}
function optimizerChange() {
    var optimizerSelect = document.getElementById("optimizer");
    var optimizerType = optimizerSelect.options[optimizerSelect.selectedIndex].value;
    var allOptional = document.getElementsByClassName("optional");
    for (var i = 0; i < allOptional.length; i++) {
        allOptional[i].style.display = 'none';
    }
    for (var i = 0; i < optimizerParams[optimizerType].length; i++) {
        document.getElementById(optimizerParams[optimizerType][i]).style.display = 'block';
    }
}


//Variable scaling
//add params
function addParam(key = null, value = null) {
    var odiv = document.createElement('div');
    odiv.className = 'param-element';
    odiv.style = 'margin-top:15px';
    var ospan1 = document.createElement('span');
    ospan1.innerText = 'key  ';
    var oinput1 = document.createElement('input');
    oinput1.className = 'key';
    oinput1.style = 'margin-right:20px';
    if (key != null) {
        oinput1.value = key;
    }
    var ospan2 = document.createElement('span');
    ospan2.innerHTML = 'value  ';
    var oinput2 = document.createElement('input');
    oinput2.className = 'value';
    oinput2.style = 'margin-right:20px';
    if (value != null) {
        oinput2.value = value;
    }
    var obutton = document.createElement('button');
    obutton.type = 'button';
    obutton.onclick = deleteParam;
    obutton.className = 'btn delete-param';
    obutton.style = 'padding: 2px 6px';
    obutton.innerText = 'DELETE';
    odiv.appendChild(ospan1);
    odiv.appendChild(oinput1);
    odiv.appendChild(ospan2);
    odiv.appendChild(oinput2);
    odiv.appendChild(obutton);
    document.getElementById("modal-content").appendChild(odiv);
}
//delete params
function deleteParam() {
    this.parentNode.parentNode.removeChild(this.parentNode)
}
//open the params box
function editParams() {
    $("#modal-params").modal('toggle');
}
//show the params
function convertParamsToShow(params) {
    document.getElementById("modal-content").innerHTML = '';
    for (var i = 0; i < params.length; i++) {
        for (var key in params[i]) {
            addParam(key, params[i][key]);
        }
    }
}
//convert params
function convertParams(value) {
    var odivs = document.getElementsByClassName('param-element');
    for (var i = 0; i < odivs.length; i++) {
        var paramKey = '{' + odivs[i].children[1].value + '}';
        var paramValue = odivs[i].children[3].value;
        if (value.indexOf(paramKey) >= 0) {
            value = value.replace(paramKey, paramValue);
        }
    }
    return value;
}


//convert model data to json data
function convertJson(jsonString) {
    var jsonObject = JSON.parse(jsonString);
    var nodeDataArray = jsonObject['nodeDataArray'];
    var attributeArray = ['loc', 'key', 'parent', 'title', 'type', 'id', 'framework', 'class_name',
        'class', 'topArray', 'leftArray', 'rightArray', 'bottomArray', 'output_shape', 'fill', 'pic_source', "necessary"]
    for (var i = 0; i < nodeDataArray.length; i++) {
        var parameters = {};
        for (var key in nodeDataArray[i]) {
            if (attributeArray.indexOf(key) >= 0) {
                continue;
            }
            parameters[key] = convertParams(nodeDataArray[i][key]);
            delete nodeDataArray[i][key];
        }
        nodeDataArray[i]['parameters'] = parameters;
    }
    // add model_parameters
    var modelParameters = {};
    var radio = document.getElementsByName("radio");
    for(var i=0;i<radio.length;i++){
        if(radio[i].checked==true) {
             modelParameters['enable_nni'] = radio[i].value;
             break;
       }
    }
    for(var i=0; i<hyperparameterElements.length; i++){
        var element = hyperparameterElements[i];
        modelParameters[element] = document.getElementsByName(element)[0].value;
    }
    modelParameters['optimizer_params'] = {};
    for (var i = 0; i < optimizerParams[modelParameters['optimizer']].length; i++) {
        var value = document.getElementsByName(optimizerParams[modelParameters['optimizer']][i])[0].value;
        if (value != '') {
            modelParameters['optimizer_params'][optimizerParams[modelParameters['optimizer']][i]] = value;
        }
    }
    jsonObject['modelParameters'] = modelParameters;
    return JSON.stringify(jsonObject);
}


//convert json data to model data
function convertNodeToShow(oldNodeDataArray) {
    var newNodeDataArray = [];
    for (var nodeIndex in oldNodeDataArray) {
        newNodeDataInfo = {};
        var oldNodeInfoInfo = oldNodeDataArray[nodeIndex];
        for (var attrIndex in oldNodeInfoInfo) {
            attrInfo = oldNodeInfoInfo[attrIndex];
            for (var key in attrInfo) {
                newNodeDataInfo[key] = attrInfo[key];
            }
        }
        newNodeDataArray.push(newNodeDataInfo);
    }
    return newNodeDataArray;
}


//show all parameters of the model
function convertModelParametersToShow(modelParameters, nniConfigParameters) {
    if(modelParameters['enable_nni']=='true'){
       enableNii();
    }
    $("#enable_nni :radio").each(function (){
       if($(this).val()==modelParameters['enable_nni']){
            $(this).attr("checked",true);
       }else{
            $(this).attr("checked",false);
       }
    });
    for(var i=0; i<hyperparameterElements.length; i++){
        var element = hyperparameterElements[i];
        document.getElementsByName(element)[0].value = modelParameters[element];
    }
    for(var i=0; i<optimizerParamsElements.length; i++){
        var element = optimizerParamsElements[i];
        document.getElementsByName(element)[0].value = modelParameters[element];
    }
    for(var i=0; i<nniConfigParamsElements.length; i++){
        var element = nniConfigParamsElements[i];
        document.getElementsByName(element)[0].value = nniConfigParameters[element];
    }

    datasetChange();
    var allOptional = document.getElementsByClassName("optional");
    for (var i = 0; i < allOptional.length; i++) {
        allOptional[i].style.display = 'none';
    }
    for (var i = 0; i < optimizerParams[modelParameters['optimizer']].length; i++) {
        document.getElementById(optimizerParams[modelParameters['optimizer']][i]).style.display = 'block';
    }
}


//check the correctness of model
function check() {
    var nodeDataArray = myDiagram.model.nodeDataArray;
    var linkDataArray = myDiagram.model.linkDataArray;

    //part1. validate the uniqueness of the title
    var titleArray = [];
    for (var index in nodeDataArray) {
        var title = nodeDataArray[index]['title'];
        if (titleArray.indexOf(title) >= 0) {
            alert('the title of ' + title + ' is exits');
            return false;
        }
        titleArray.push(title);
    }

    //part2. validate the Continuity of subGraphInput order and the uniqueness of the variable-name
    linksInfo = {};
    for (var index in linkDataArray) {
        var from = linkDataArray[index]['from'];
        var to = linkDataArray[index]['to'];
        if (Object.keys(linksInfo).indexOf(String(to)) < 0) {
            linksInfo[linkDataArray[index]['to']] = [];
        }
        linksInfo[linkDataArray[index]['to']].push(from);
    }

    inputArray = {};
    outputArray = {};
    for (var index in nodeDataArray) {
        var node = nodeDataArray[index];
        var type = node['type'];
        var key = node['key'];
        if ('KerasModelInput' === type || 'SubGraphIn' === type) {
            inputArray[key] = node;
        }
        if ('KerasModelOutput' === type || 'SubGraphOut' === type) {
            outputArray[key] = node;
        }
    }

    for (var key in outputArray) {
        var queue = [];
        queue.push(key);
        var nodeArray = {};
        while (queue.length != 0) {
            var nodeKey = queue.pop();
            if (Object.keys(linksInfo).indexOf(String(nodeKey)) >= 0) {
                for (var index in linksInfo[nodeKey]) {
                    queue.push(linksInfo[nodeKey][index]);
                }
            } else {
                if (!(nodeKey in nodeArray)) {
                    nodeArray[nodeKey] = inputArray[nodeKey];
                }
            }
        }

        var orderArray = [];
        var variableNameArray = [];
        for (var index in nodeArray) {
            var order = nodeArray[index]['order'];
            var variableName = nodeArray[index]['variable-name'];
            orderArray.push(order);
            if (variableNameArray.indexOf(variableName) >= 0) {
                alert(nodeArray[index]['title'] + " variable-name repeat!");
                return false;
            }
            variableNameArray.push(variableName);
        }

        for (var index = 0; index < orderArray.length; index++) {
            if (orderArray.indexOf(String(index)) < 0) {
                alert(outputArray[key]['title'] + " inputs order Inconsistent!");
                return false;
            }
        }
    }
    return true;
}


//convert model data to save
function convertToSave(model_json) {
    //parse json
    var model_json = JSON.parse(model_json);
    //convert nodeDataArray
    var newNodeDataArray = [];
    var oldNodeDataArray = model_json['nodeDataArray'];
    for (var i = 0; i < oldNodeDataArray.length; i++) {
        oldNodeDataInfo = oldNodeDataArray[i];
        newNodeDataInfo = [];
        for (var key in oldNodeDataInfo) {
            var paramInfo = {};
            paramInfo[key] = oldNodeDataInfo[key];
            newNodeDataInfo.push(paramInfo);
        }
        newNodeDataArray.push(newNodeDataInfo);
    }
    model_json['nodeDataArray'] = newNodeDataArray;
    //convert params
    var paramsDataArray = [];
    var odivs = document.getElementsByClassName('param-element');
    for (var i = 0; i < odivs.length; i++) {
        var param = {};
        var key = odivs[i].children[1].value;
        var value = odivs[i].children[3].value;
        param[key] = value;
        paramsDataArray.push(param);
    }
    model_json['params'] = paramsDataArray;
    //convert model-parameters
    var modelParameters = {};
    var radio = document.getElementsByName("radio");
    for(var i=0;i<radio.length;i++){
        if(radio[i].checked==true) {
             modelParameters['enable_nni'] = radio[i].value;
             break;
       }
    }
    for(var i=0; i<hyperparameterElements.length; i++){
        var element = hyperparameterElements[i];
        modelParameters[element] = document.getElementsByName(element)[0].value;
    }
    for(var i=0; i<optimizerParamsElements.length; i++){
        var element = optimizerParamsElements[i];
        modelParameters[element] = document.getElementsByName(element)[0].value;
    }
    model_json['modelParameters'] = modelParameters;

    var nniConfigParameters = {};
    for(var i=0; i<nniConfigParamsElements.length; i++){
        var element = nniConfigParamsElements[i];
        nniConfigParameters[element] = document.getElementsByName(element)[0].value;
    }
    model_json['nniConfigParameters'] = nniConfigParameters;
    //stringfy json
    model_json = JSON.stringify(model_json);
    return model_json;
}


//copy generater code
function copyText() {
    var copy_code = document.getElementById("codeTextArea");
    copy_code.select();
    try {
        var successful = document.execCommand('Copy');
        var msg = successful ? 'successful' : 'unsuccessful';
        alert('Copied!');
    } catch (err) {
        alert('Failed!');
    }
}

//stop the jupyter
function stop_jupyter() {
    $.getJSON('/stop_jupyter', {}, function (data) {
        alert(data.page_code);
    });
}

function init() {
    if (window.goSamples) goSamples();
    var $ = go.GraphObject.make;
    go.licenseKey = "73f041e5b61c28c702d90776423d6bf919a428639b841ba00c0713f7ef083f1d779cba7106d789c287f84" +
        "8fb1d7e97898dc56e7ec04f013ce738868913e6d4a9e63323b2100917dea35024c79ce83aa4fe2b24f396e627f6d97a85" +
        "f2b9fa939a0ce1a3d048cc0bb92c7f0333532da74fe7ac8c79ae059947633f98a6fab9ac4df86d25968ee202d8e959238" +
        "ebeffb05d77701fc03ee275";
    myDiagram =
        $(go.Diagram, "myDiagramDiv",
            {"undoManager.isEnabled": true});

    myDiagram.addDiagramListener("ExternalObjectsDropped", function (e) {
        setNumber();
    });
    myDiagram.addDiagramListener("ClipboardPasted", function (e) {
        setNumber();
    });

    myDiagram.addDiagramListener("Modified", function (e) {
        var idx = document.title.indexOf("*");
        if (myDiagram.isModified) {
            if (idx < 0) document.title += "*";
        } else {
            if (idx >= 0) document.title = document.title.substr(0, idx);
        }
    });

    myDiagram.addDiagramListener("ChangedSelection", function (diagramEvent) {
        infoDraggable.style.display = 'inline-block';
        document.getElementById('myDiagramDiv').className = 'row col-xs-8';
    });

    myDiagram.addDiagramListener("BackgroundSingleClicked", function (diagramEvent) {
        infoDraggable.style.display = 'none';
        document.getElementById('myDiagramDiv').className = 'row col-xs-10';
    });

    function makeButton(text, action, visiblePredicate) {
        return $("ContextMenuButton",
            $(go.TextBlock, text),
            {click: action},
            visiblePredicate ? new go.Binding("visible", "", function (o, e) {
                return o.diagram ? visiblePredicate(o, e) : false;
            }).ofObject() : {});
    }

    var nodeMenu =
        $("ContextMenu",
            makeButton("Copy",
                function (e, obj) {
                    e.diagram.commandHandler.copySelection();
                }),
            makeButton("Delete",
                function (e, obj) {
                    e.diagram.commandHandler.deleteSelection();
                }),
            $(go.Shape, "LineH", {strokeWidth: 2, height: 1, stretch: go.GraphObject.Horizontal}),
            makeButton("Add top port",
                function (e, obj) {
                    addPort("top");
                }),
            makeButton("Add bottom port",
                function (e, obj) {
                    addPort("bottom");
                })
        );

    var portSize = new go.Size(8, 8);

    var portMenu =
        $("ContextMenu",
            makeButton("Remove port",
                function (e, obj) {
                    removePort(obj.part.adornedObject);
                }),
        );

    myDiagram.nodeTemplate =
        $(go.Node, "Table",
            {
                locationObjectName: "BODY",
                locationSpot: go.Spot.Center,
                selectionObjectName: "BODY",
                contextMenu: nodeMenu
            },
            new go.Binding("location", "loc", go.Point.parse).makeTwoWay(go.Point.stringify),

            $(go.Panel, "Auto",
                {
                    row: 1, column: 1, name: "BODY",
                    stretch: go.GraphObject.Fill
                },
                $(go.Shape, "Rectangle",
                    {
                        fill: '#ff5768', stroke: null, strokeWidth: 0,
                        minSize: new go.Size(160, 60),
                    },
                    new go.Binding('fill', 'fill')),
                $(go.TextBlock,
                    {
                        margin: 10,
                        textAlign: "center",
                        font: " 10pt Sans-Serif",
                        stroke: "white",
                        alignment: go.Spot.Top,
                        verticalAlignment: go.Spot.Top,
                        editable: false
                    },
                    new go.Binding("text", "title").makeTwoWay()
                ),
                $(go.TextBlock,
                    {
                        margin: 10,
                        textAlign: "center",
                        font: " 10pt Sans-Serif",
                        stroke: "white",
                        alignment: go.Spot.Bottom,
                        verticalAlignment: go.Spot.Bottom,
                        editable: false
                    },
                    new go.Binding("text", "output_shape").makeTwoWay()
                )
            ),

            $(go.Panel, "Vertical",
                new go.Binding("itemArray", "leftArray"),
                {
                    row: 1, column: 0,
                    itemTemplate:
                        $(go.Panel,
                            {
                                _side: "left",
                                fromSpot: go.Spot.Left, toSpot: go.Spot.Left,
                                fromLinkable: true, toLinkable: true, cursor: "pointer",
                                contextMenu: portMenu
                            },
                            new go.Binding("portId", "portId"),
                            $(go.Shape, "Rectangle",
                                {
                                    stroke: null, strokeWidth: 0,
                                    desiredSize: portSize,
                                    margin: new go.Margin(1, 0)
                                },
                                new go.Binding("fill", "portColor"))
                        )
                }
            ),

            $(go.Panel, "Horizontal",
                new go.Binding("itemArray", "topArray"),
                {
                    row: 0, column: 1,
                    itemTemplate:
                        $(go.Panel,
                            {
                                _side: "top",
                                fromSpot: go.Spot.Top, toSpot: go.Spot.Top,
                                fromLinkable: true, toLinkable: true, cursor: "pointer",
                                contextMenu: portMenu
                            },
                            new go.Binding("portId", "portId"),
                            $(go.Shape, "Rectangle",
                                {
                                    stroke: null, strokeWidth: 0,
                                    desiredSize: portSize,
                                    margin: new go.Margin(0, 1)
                                },
                                new go.Binding("fill", "portColor"))
                        )
                }
            ),

            $(go.Panel, "Vertical",
                new go.Binding("itemArray", "rightArray"),
                {
                    row: 1, column: 2,
                    itemTemplate:
                        $(go.Panel,
                            {
                                _side: "right",
                                fromSpot: go.Spot.Right, toSpot: go.Spot.Right,
                                fromLinkable: true, toLinkable: true, cursor: "pointer",
                                contextMenu: portMenu
                            },
                            new go.Binding("portId", "portId"),
                            $(go.Shape, "Rectangle",
                                {
                                    stroke: null, strokeWidth: 0,
                                    desiredSize: portSize,
                                    margin: new go.Margin(1, 0)
                                },
                                new go.Binding("fill", "portColor"))
                        )
                }
            ),

            $(go.Panel, "Horizontal",
                new go.Binding("itemArray", "bottomArray"),
                {
                    row: 2, column: 1,
                    itemTemplate:
                        $(go.Panel,
                            {
                                _side: "bottom",
                                fromSpot: go.Spot.Bottom, toSpot: go.Spot.Bottom,
                                fromLinkable: true, toLinkable: true, cursor: "pointer",
                                contextMenu: portMenu
                            },
                            new go.Binding("portId", "portId"),
                            $(go.Shape, "Rectangle",
                                {
                                    stroke: null, strokeWidth: 0,
                                    desiredSize: portSize,
                                    margin: new go.Margin(0, 1)
                                },
                                new go.Binding("fill", "portColor"))
                        )
                }
            )
        );

    myDiagram.linkTemplate =
        $(go.Link,
            {
                routing: go.Link.AvoidsNodes,
                curve: go.Link.JumpOver,
                corner: 5, toShortLength: 4,
                relinkableFrom: true,
                relinkableTo: true,
                reshapable: true,
                resegmentable: true,
                mouseEnter: function (e, link) {
                    link.findObject("HIGHLIGHT").stroke = "rgba(30,144,255,0.2)";
                },
                mouseLeave: function (e, link) {
                    link.findObject("HIGHLIGHT").stroke = "transparent";
                },
                selectionAdorned: false, selectable: true
            },
            new go.Binding("points").makeTwoWay(),
            $(go.Shape,
                {isPanelMain: true, strokeWidth: 8, stroke: "transparent", name: "HIGHLIGHT"}),
            $(go.Shape,
                {isPanelMain: true, stroke: "gray", strokeWidth: 2,},
                new go.Binding("stroke", "isSelected", function (sel) {
                    return sel ? "dodgerblue" : "gray";
                }).ofObject()),
            $(go.Shape,
                {toArrow: "standard", strokeWidth: 0, fill: "gray"})
        );

    myDiagram.contextMenu =
        $("ContextMenu",
            makeButton("Paste",
                function (e, obj) {
                    e.diagram.commandHandler.pasteSelection(e.diagram.lastInput.documentPoint);
                },
                function (o) {
                    return o.diagram.commandHandler.canPasteSelection();
                }),
            makeButton("Undo",
                function (e, obj) {
                    e.diagram.commandHandler.undo();
                },
                function (o) {
                    return o.diagram.commandHandler.canUndo();
                }),
            makeButton("Redo",
                function (e, obj) {
                    e.diagram.commandHandler.redo();
                },
                function (o) {
                    return o.diagram.commandHandler.canRedo();
                })
        );

    myPalette =
        $(go.Palette, "myPaletteDiv",
            {
                layout:
                    $(go.TreeLayout,
                        {
                            alignment: go.TreeLayout.AlignmentStart,
                            angle: 0,
                            compaction: go.TreeLayout.CompactionNone,
                            layerSpacing: 16,
                            layerSpacingParentOverlap: 1,
                            nodeIndentPastParent: 1.0,
                            nodeSpacing: 0,
                            setsPortSpot: false,
                            setsChildPortSpot: false,
                        })
            });

    myPalette.nodeTemplate =
        $(go.Node,
            {
                selectionAdorned: false,
                doubleClick: function (e, node) {
                    var cmd = myPalette.commandHandler;
                    e.handled = true;
                    if (node.isTreeExpanded) {
                        cmd.collapseTree(node);
                    } else {
                        cmd.expandTree(node);
                    }
                }
            },
            $(go.Panel, "Horizontal",
                {position: new go.Point(18, 0)},
                new go.Binding("background", "isSelected", function (s) {
                    return (s ? "lightblue" : "#ecf0f5");
                }).ofObject(),
                $(go.Picture,
                    {
                        width: 20, height: 20,
                        margin: new go.Margin(0, 4, 0, 0),
                        imageStretch: go.GraphObject.Uniform,
                        source: "/assets/img/goJs/document.svg",
                    },
                    new go.Binding("source", "pic_source")),
                $(go.TextBlock,
                    {font: '12pt Source Sans Pro,sans-serif'},
                    new go.Binding("text", "title")
                )
            )
        );

    myPalette.linkTemplate = $(go.Link);
    myPalette.model = new go.TreeModel(nodeDataArray);

    myPalette.addDiagramListener("InitialLayoutCompleted", function (diagramEvent) {
        var pdrag = document.getElementById("paletteDraggable");
        var palette = diagramEvent.diagram;
    });

    collapseParentNode();
    load();

}

function collapseParentNode() {
    for (var i = 1; i < 100; i++) {
        var node = myPalette.findNodeForKey(i);
        if (node) {
            node.collapseTree();
        } else {
            break;
        }
    }
}

function CustomLink() {
    go.Link.call(this);
};

go.Diagram.inherit(CustomLink, go.Link);

CustomLink.prototype.findSidePortIndexAndCount = function (node, port) {
    var nodedata = node.data;
    if (nodedata !== null) {
        var portdata = port.data;
        var side = port._side;
        var arr = nodedata[side + "Array"];
        var len = arr.length;
        for (var i = 0; i < len; i++) {
            if (arr[i] === portdata) return [i, len];
        }
    }
    return [-1, len];
};

CustomLink.prototype.computeEndSegmentLength = function (node, port, spot, from) {
    var esl = go.Link.prototype.computeEndSegmentLength.call(this, node, port, spot, from);
    var other = this.getOtherPort(port);
    if (port !== null && other !== null) {
        var thispt = port.getDocumentPoint(this.computeSpot(from));
        var otherpt = other.getDocumentPoint(this.computeSpot(!from));
        if (Math.abs(thispt.x - otherpt.x) > 20 || Math.abs(thispt.y - otherpt.y) > 20) {
            var info = this.findSidePortIndexAndCount(node, port);
            var idx = info[0];
            var count = info[1];
            if (port._side == "top" || port._side == "bottom") {
                if (otherpt.x < thispt.x) {
                    return esl + 4 + idx * 8;
                } else {
                    return esl + (count - idx - 1) * 8;
                }
            } else {  // left or right
                if (otherpt.y < thispt.y) {
                    return esl + 4 + idx * 8;
                } else {
                    return esl + (count - idx - 1) * 8;
                }
            }
        }
    }
    return esl;
};

CustomLink.prototype.hasCurviness = function () {
    if (isNaN(this.curviness)) return true;
    return go.Link.prototype.hasCurviness.call(this);
};

CustomLink.prototype.computeCurviness = function () {
    if (isNaN(this.curviness)) {
        var fromnode = this.fromNode;
        var fromport = this.fromPort;
        var fromspot = this.computeSpot(true);
        var frompt = fromport.getDocumentPoint(fromspot);
        var tonode = this.toNode;
        var toport = this.toPort;
        var tospot = this.computeSpot(false);
        var topt = toport.getDocumentPoint(tospot);
        if (Math.abs(frompt.x - topt.x) > 20 || Math.abs(frompt.y - topt.y) > 20) {
            if ((fromspot.equals(go.Spot.Left) || fromspot.equals(go.Spot.Right)) &&
                (tospot.equals(go.Spot.Left) || tospot.equals(go.Spot.Right))) {
                var fromseglen = this.computeEndSegmentLength(fromnode, fromport, fromspot, true);
                var toseglen = this.computeEndSegmentLength(tonode, toport, tospot, false);
                var c = (fromseglen - toseglen) / 2;
                if (frompt.x + fromseglen >= topt.x - toseglen) {
                    if (frompt.y < topt.y) return c;
                    if (frompt.y > topt.y) return -c;
                }
            } else if ((fromspot.equals(go.Spot.Top) || fromspot.equals(go.Spot.Bottom)) &&
                (tospot.equals(go.Spot.Top) || tospot.equals(go.Spot.Bottom))) {
                var fromseglen = this.computeEndSegmentLength(fromnode, fromport, fromspot, true);
                var toseglen = this.computeEndSegmentLength(tonode, toport, tospot, false);
                var c = (fromseglen - toseglen) / 2;
                if (frompt.x + fromseglen >= topt.x - toseglen) {
                    if (frompt.y < topt.y) return c;
                    if (frompt.y > topt.y) return -c;
                }
            }
        }
    }
    return go.Link.prototype.computeCurviness.call(this);
};

function addPort(side) {
    myDiagram.startTransaction("addPort");
    myDiagram.selection.each(function (node) {
        if (!(node instanceof go.Node)) return;
        var i = 0;
        while (node.findPort(side + i.toString()) !== node) i++;
        var name = side + i.toString();
        var arr = node.data[side + "Array"];
        if (arr) {
            var newportdata = {
                portId: name,
                portColor: go.Brush.randomColor()
            };
            myDiagram.model.insertArrayItem(arr, -1, newportdata);
        }
    });
    myDiagram.commitTransaction("addPort");
}

function removePort(port) {
    myDiagram.startTransaction("removePort");
    var pid = port.portId;
    var arr = port.panel.itemArray;
    for (var i = 0; i < arr.length; i++) {
        if (arr[i].portId === pid) {
            myDiagram.model.removeArrayItem(arr, i);
            break;
        }
    }
    myDiagram.commitTransaction("removePort");
}

$.getJSON('/getNodeData', function (data) {
    nodeDataArray = convertNodeToShow(data.result);
    getJsonData();
});
