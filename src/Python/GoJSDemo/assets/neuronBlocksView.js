first_open = true
// convert different step
function openData() {
    dataButton.style.background = 'lightblue';
    parameterButton.style.background = '';
    modelButton.style.background = '';
    parameterContent.style.display = 'none';
    mainContent.style.display = 'none';
    dataContent.style.display = 'block';
};

function openModel() {
    modelButton.style.background = 'lightblue';
    parameterButton.style.background = '';
    dataButton.style.background = '';
    parameterContent.style.display = 'none';
    dataContent.style.display = 'none';
    if(first_open){
        init();
        first_open = false
    }
    mainContent.style.display = 'block';
};

function openParameter() {
    parameterButton.style.background = 'lightblue';
    modelButton.style.background = '';
    dataButton.style.background = '';
    mainContent.style.display = 'none';
    dataContent.style.display = 'none';
    parameterContent.style.display = 'block';
};

//order by every elements
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

//different params
var modelParametersKey = ['license', 'tool_version', 'model_description', 'use_cache', 'dataset_type', 'tagging_scheme', 'data_paths', 'pretrained_emb_type',
    'pretrained_emb_binary_or_text', 'file_with_col_header', 'add_start_end_for_seq',
    'involve_all_words_in_pretrained_emb', 'file_header', 'predict_file_header', 'model_inputs', 'target',
    'embedding_conf', 'save_base_dir', 'model_name', 'train_log_name', 'test_log_name', 'predict_log_name',
    'predict_fields', 'predict_output_name', 'cache_dir', 'losses_type', 'losses_conf', 'losses_inputs',
    'metrics', 'vocabulary', 'optimizer', 'lr_decay', 'minimum_lr', 'epoch_start_lr_decay', 'use_gpu',
    'batch_size', 'batch_num_to_show_results', 'max_epoch', 'valid_times_per_epoch', 'text_preprocessing',
    'max_lengths'];
var basicParametersKey = ['license', 'tool_version', 'model_description'];
var inputParametersKey = ['use_cache', 'dataset_type', 'tagging_scheme', 'data_paths', 'pretrained_emb_type',
    'pretrained_emb_binary_or_text', 'file_with_col_header', 'add_start_end_for_seq',
    'involve_all_words_in_pretrained_emb', 'file_header', 'predict_file_header', 'model_inputs', 'target',
    'embedding_conf'];
var outputParametersKey = ['save_base_dir', 'model_name', 'train_log_name', 'test_log_name', 'predict_log_name',
    'predict_fields', 'predict_output_name', 'cache_dir'];
var trainingParametersKey = ['vocabulary', 'optimizer', 'lr_decay', 'minimum_lr', 'epoch_start_lr_decay', 'use_gpu',
    'batch_size', 'batch_num_to_show_results', 'max_epoch', 'valid_times_per_epoch', 'text_preprocessing',
    'max_lengths'];
var lossParametersKey = ['losses_type', 'losses_conf', 'losses_inputs'];
var metricsParametersKey = 'metrics';

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

//convert model data to json date
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
    //convert model-parameters
    var modelParameters = {};
    for (var i = 0; i < modelParametersKey.length; i++) {
        var key = modelParametersKey[i];
        modelParameters[key] = document.getElementsByName(key)[0].value;
    }
    model_json['modelParameters'] = modelParameters;
    //stringfy json
    model_json = JSON.stringify(model_json);
    return model_json;
}

//show all parameters
function convertModelParametersToShow(modelParameters) {
    for (var i = 0; i < modelParametersKey.length; i++) {
        var key = modelParametersKey[i];
        document.getElementsByName(key)[0].value = modelParameters[key];
    }
}

//filter and collect the params of elements
function filterParams(nodeData) {
    var attributeArray = ['loc', 'key', 'parent', 'title', 'type', 'id', 'framework', 'class_name',
        'class', 'topArray', 'leftArray', 'rightArray', 'bottomArray', 'output_shape', 'fill', 'pic_source', "necessary"];
    var parameters = {};
    for (var key in nodeData) {
        if(key.indexOf('__') == 0 || attributeArray.indexOf(key) >= 0){
            continue;
        }
        try {
            if(nodeData[key] == 'True' || nodeData[key] == 'False'){
                parameters[key] = JSON.parse(nodeData[key].toLowerCase());
            }else{
                parameters[key] = JSON.parse(nodeData[key]);
            }
        } catch (e) {
            parameters[key] = nodeData[key];
        }
    }
    return parameters;
}

//preview
function preview() {
    var json_content = {};
    //basic layer
    for (var i = 0; i < basicParametersKey.length; i++) {
        var key = basicParametersKey[i];
        var value = document.getElementsByName(key)[0].value;
        json_content[key] = value;
    }
    //input layer
    var input_data = {};
    for (var i = 0; i < inputParametersKey.length; i++) {
        var key = inputParametersKey[i];
        var value = document.getElementsByName(key)[0].value;
        if(key == 'embedding_conf' || value == ''){
            continue;
        }
        try {
            input_data[key] = JSON.parse(value);
        } catch (e) {
            input_data[key] = value;
        }
    }
    json_content['inputs'] = input_data;
    //output layer
    var output_data = {};
    for (var i = 0; i < outputParametersKey.length; i++) {
        var key = outputParametersKey[i];
        var value = document.getElementsByName(key)[0].value;
        try {
            output_data[key] = JSON.parse(value);
        } catch (e) {
            output_data[key] = value;
        }
    }
    json_content['outputs'] = output_data;
    //training layer
    var training_data = {};
    for (var i = 0; i < trainingParametersKey.length; i++) {
        var key = trainingParametersKey[i];
        var value = document.getElementsByName(key)[0].value;
        if(value == ''){
            continue;
        }
        try {
            training_data[key] = JSON.parse(value);
        } catch (e) {
            training_data[key] = value;
        }
    }
    json_content['training_params'] = training_data;
    //loss layer
    var losses_data = {};
    for (var i = 0; i < lossParametersKey.length; i++) {
        var key = lossParametersKey[i];
        var value = document.getElementsByName(key)[0].value;
        try {
            losses_data[key.replace('losses_', '')] = JSON.parse(value);
        } catch (e) {
            losses_data[key.replace('losses_', '')] = value;
        }
    }
    var losses_array = [losses_data];
    var loss_data = {
        'losses': losses_array
    };
    json_content['loss'] = loss_data;
    //metrics layer
    var value = document.getElementsByName(metricsParametersKey)[0].value;
    try {
        metricsValue = JSON.parse(value);
    } catch (e) {
        metricsValue = value;
    }
    json_content['metrics'] = metricsValue;
    //architecture layer
    var nodeDataArray = myDiagram.model.nodeDataArray;
    var output = null;
    var nodeDataMap = {};
    for (var i = 0; i < nodeDataArray.length; i++) {
        if (nodeDataArray[i]['type'] == 'NeuronOutput') {
            output = nodeDataArray[i];
        }
        nodeDataMap[nodeDataArray[i]['key']] = nodeDataArray[i];
    }
    var linkDataArray = myDiagram.model.linkDataArray;
    var linkDataMap = {};
    for (var i = 0; i < linkDataArray.length; i++) {
        if (!linkDataMap[linkDataArray[i]['to']]) {
            linkDataMap[linkDataArray[i]['to']] = [];
        }
        linkDataMap[linkDataArray[i]['to']].push(nodeDataMap[linkDataArray[i]['from']]);
    }
    var architecture = {};
    if (output == null) {
        alert('must has a NeuronOutput block!');
        return false;
    }
    var nodeList = [];
    nodeList.push(output);
    var start = 0;
    var end = 1;
    while (end - start > 0) {
        var node = nodeList[start];
        var fromList = linkDataMap[node['key']];
        var inputsData = [];
        for (var i = 0; i < fromList.length; i++) {
            var fromNode = fromList[i];
            if (fromNode['type'] == 'NeuronInput') {
                inputsData.push(fromNode['name']);
                continue;
            } else {
                inputsData.push(fromNode['type'] + '_' + fromNode['key']);
                var data = {};
                if (node['type'] == 'NeuronOutput') {
                    data['output_layer_flag'] = true;
                    data['layer_id'] = 'output';
                } else {
                    data['layer_id'] = fromNode['type'] + '_' + fromNode['key'];
                }
                data['layer'] = fromNode['type'];
                data['conf'] = filterParams(fromList[i]);
                architecture[fromList[i]['key']] = data;
                nodeList.push(fromList[i]);
                end++;
            }
        }
        if (node['type'] != 'NeuronOutput') {
            architecture[node['key']]['inputs'] = inputsData;
        }
        start++;
    }
    var architecture_data = [];
    for (var key in architecture) {
        architecture_data.unshift(architecture[key]);
    }
    var embedding_conf_value;
    try {
        embedding_conf_value = JSON.parse(document.getElementsByName("embedding_conf")[0].value);
    } catch (e) {
        alert(1)
        embedding_conf_value = document.getElementsByName("embedding_conf")[0].value;
    }
    var embedding_layer = {
        'layer': "Embedding",
        "conf": embedding_conf_value,
    };
    architecture_data.unshift(embedding_layer);
    json_content['architecture'] = architecture_data;

    $("#modal-default").modal('toggle');
    var page_code = document.getElementById("codeTextArea");
    page_code.value = JSON.stringify(json_content);
}

//copy python code
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



//goJs code
var nodeDataArray = [];
var modelData = {};
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
            makeButton("Swap order",
                function (e, obj) {
                    swapOrder(obj.part.adornedObject);
                }),
            makeButton("Remove port",
                function (e, obj) {
                    removePort(obj.part.adornedObject);
                }),
            makeButton("Change color",
                function (e, obj) {
                    changeColor(obj.part.adornedObject);
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

function swapOrder(port) {
    var arr = port.panel.itemArray;
    if (arr.length >= 2) {
        for (var i = 0; i < arr.length; i++) {
            if (arr[i].portId === port.portId) {
                myDiagram.startTransaction("swap ports");
                if (i >= arr.length - 1) i--;
                var newarr = arr.slice(0);
                newarr[i] = arr[i + 1];
                newarr[i + 1] = arr[i];
                myDiagram.model.setDataProperty(port.part.data, port._side + "Array", newarr);
                myDiagram.commitTransaction("swap ports");
                break;
            }
        }
    }
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

function changeColor(port) {
    myDiagram.startTransaction("colorPort");
    var data = port.data;
    myDiagram.model.setDataProperty(data, "portColor", go.Brush.randomColor());
    myDiagram.commitTransaction("colorPort");
}

$.getJSON('/getNeuronBlocksData', function (data) {
    nodeDataArray = convertNodeToShow(data.result);
    getJsonData();
});
