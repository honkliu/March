// Copyright (c) 2010 The Chromium Authors. All rights reserved.  
// Use of this source code is governed by a BSD-style license that can be  
// found in the LICENSE file.  
  
'use strict';

const $ = require('./jquery-3.1.0.js');
const request = require('request');
const uuidv4 = require('uuid/v4');
chrome.browserAction.setBadgeText({text: 'ON'});
chrome.browserAction.setBadgeBackgroundColor({color: 'teal'});

// var key_var = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY';
// if (!process.env[key_var]) {
//     throw new Error('Please set/export the following environment variable: ' + key_var);
// }
// var subscriptionKey = process.env[key_var];
// var endpoint_var = 'TRANSLATOR_TEXT_ENDPOINT';
// if (!process.env[endpoint_var]) {
//     throw new Error('Please set/export the following environment variable: ' + endpoint_var);
// }
// var endpoint = process.env[endpoint_var];
const subscriptionKey_text = '58cd5b01f6004c7f9642ca8b7b694170';
const subscriptionKey_speech = 'ea80d576254f4df6860cc730a1981358';


const translateToText = (text) => {
  const BaseRequestUrl = `https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from=en&to=zh-Hans`;
  fetch(BaseRequestUrl, {
    method: 'post',
    headers: {
      'Ocp-Apim-Subscription-Key': subscriptionKey_text,
      'Content-type': 'application/json',
      'X-ClientTraceId': uuidv4().toString()
    },
    body: `[{"Text": "${text}"}]`,
  }).then((res) => {
    if (!res.ok) {
      throw Error('Failed to translate text');
    }
    res.json().then((data) => {
      console.log(data[0].translations[0].text);
      sendMessageToContentScript({type: 'toText', value: data[0].translations[0].text}, function(response) {
        console.log(response, 44);
      });
    }).catch(() => {
      throw Error('translate failed');
    });
  }).catch((error)=>{});

};


const translateToSpeech = (text) => {
  console.log('translateToSpeech');
  $.ajax({
    url: `http://localhost:2000/texttospeech?text=${text}`,
    type: "post",
    data: text,
    dataType: 'text',
    success: function(data) {
      console.log(data);
      sendMessageToContentScript({type: 'toSpeech', value: data}, function(response) {
        console.log(response, 64);
      });
      },
    })
}
//菜单项
const options = [
  {
    id: 'translatetotext',
    title: 'translate to text',
    type: 'normal',
    contexts: ['selection'],
    visible: true,
  },{
    id: 'translatetospeech',
    title: 'translate to speech',
    type: 'normal',
    contexts: ['selection'],
    visible: true,
  }
];
//添加右键菜单项
chrome.runtime.onInstalled.addListener(function() {
  options.forEach((item, index) => {
    chrome.contextMenus.create({
      id: item.id,
      title: item.title,
      contexts: item.contexts,
      visible: item.visible,
    })
  });
  chrome.contextMenus.onClicked.addListener((prama) => {
    console.log(prama);
    if (prama.menuItemId === 'translatetotext') {
      translateToText(prama.selectionText);
    } else if (prama.menuItemId === 'translatetospeech') {
      translateToSpeech(prama.selectionText);
    }
  })
})
console.log('background');
// 发送消息
// function sendMessageToContentScript(message, callback) {
//   chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
//     chrome.tabs.sendMessage(tabs[0].id, message, function(response) {
//         if(callback) {
//           callback(response)
//         };
//     });
//   });
// }

function sendMessageToContentScript(message, callback) {
  chrome.tabs.query(
    {active: true, currentWindow: true},
    function (tabs) {
        var port = chrome.tabs.connect(//建立通道
            tabs[0].id,
            {name: message.type}//通道名称
        );
        port.postMessage({value: message.value});//向通道中发送消息
        port.onMessage.addListener(function (msg) {//这里同时利用通道建立监听消息，可以监听对方返回的消息
          console.log(msg);
        });
    });
}