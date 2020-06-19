
const translateDiv = document.createElement('div');
const translateResult = document.createElement('div');
const cancleButton = document.createElement('div');

const audio = document.createElement('audio');
audio.controls = true;
audio.id = 'audio';

translateDiv.id = 'translation';
translateResult.id = 'translateResult';
cancleButton.id = 'cancleButton';
cancleButton.innerHTML = 'Cancle';
translateDiv.appendChild(translateResult);
translateDiv.appendChild(audio);
translateDiv.appendChild(cancleButton);
document.body.appendChild(translateDiv);

// 鼠标按下事件
translateDiv.onmousedown = function(e) {
  const x = e.clientX;
  const y = e.clientY
  const left = parseInt(getComputedStyle(translateDiv)["left"]);
  const top = parseInt(getComputedStyle(translateDiv)["top"]);
  document.onmousemove = function(e) {
    const m_x = e.clientX;
    const m_y = e.clientY;
    translateDiv.style.left = m_x - x + left + "px";
    translateDiv.style.top = m_y - y + top + "px";
  }
}

//取消鼠标移动事件
document.onmouseup = function() {
  document.onmousemove = null;
} 
//取消按钮
cancleButton.onclick = () => {
  translateDiv.style.display = 'none';
}

// chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
//   console.log(request);
//   if (request.type === 'toText') {
//     translateDiv.style.display = 'block';
//     translateResult.style.display = 'block';
//     audio.style.display = 'none';
//     translateResult.innerHTML = request.value;
//     sendResponse({text: '转文字成功'});
//   } else if (request.type === 'toSpeech') {
//     translateDiv.style.display = 'block';
//     translateResult.style.display = 'none';
//     audio.style.display = 'block';
//     // audio.src = request.value;
//     audio.src = "https://qdcu01.baidupcs.com/file/d7ef45a51d7a3c4b47b15d034cdfaf96?bkt=en-6766f9da69592c12bb2c7d91a0f901fa29bd62ef4e0987e4d6c9f403a2e5599c60e2c6b8333a8236e33624c0415c8ae1249ac802d44dfb93d8ad14efb4de841b&fid=2117437711-250528-401418009448741&time=1572320105&sign=FDTAXGERLQBHSKfW-DCb740ccc5511e5e8fedcff06b081203-D7i6Ckgxk41i%2BdpejPaS5WTTeic%3D&to=65&size=1090604&sta_dx=1090604&sta_cs=1&sta_ft=mp3&sta_ct=0&sta_mt=0&fm2=MH%2CYangquan%2CAnywhere%2C%2CNone%2Cany&ctime=1572319467&mtime=1572319467&resv0=cdnback&resv1=0&resv2=rlim&resv3=5&resv4=1090604&vuk=2117437711&iv=0&htype=&randtype=&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-02895e0652184dc39c52b381582350dbfaf3590299ce4b74bd47909e9e7a0b470e39b0418692076aa08b5c8c0054e9c6c8abe51fcfd7e680305a5e1275657320&sl=68616270&expires=8h&rt=sh&r=602439256&vbdid=2939104332&fin=TTSOutput.mp3&fn=TTSOutput.mp3&rtype=1&dp-logid=6992234819097515635&dp-callid=0.1&hps=1&tsl=200&csl=200&csign=HMiavZmK%2B%2BVwV1XQ0oUsBZvJjhU%3D&so=0&ut=6&uter=4&serv=0&uc=738096701&ti=648eaef5c3fa81d1abb7b6c60806c5e1a5390508c7e5871f&reqlabel=250528_f&by=themis";
//     audio.play();
//     sendResponse({text: '转语音成功'});
//   }
//   console.log(58);
//   translateDiv.style.display = 'block';

// });
// console.log('----');

chrome.runtime.onConnect.addListener(function(port) {
  console.log(port);
  if(port.name === 'toText') {
      port.onMessage.addListener(function(msg) {
        console.log('收到翻译文本：', msg);
        translateDiv.style.display = 'block';
        translateResult.style.display = 'block';
        audio.style.display = 'none';
        translateResult.innerHTML = msg.value;
        port.postMessage({answer: 'to text succeed', status: 'ok'});
      });
  } else if(port.name === 'toSpeech') {
      port.onMessage.addListener(function(msg) {
        console.log('收到语音文件地址：', msg);
        translateDiv.style.display = 'block';
        translateResult.style.display = 'none';
        audio.style.display = 'block';
        audio.src = "https://qdcu01.baidupcs.com/file/d7ef45a51d7a3c4b47b15d034cdfaf96?bkt=en-6766f9da69592c12bb2c7d91a0f901fa29bd62ef4e0987e4d6c9f403a2e5599c60e2c6b8333a8236e33624c0415c8ae1249ac802d44dfb93d8ad14efb4de841b&fid=2117437711-250528-401418009448741&time=1572320105&sign=FDTAXGERLQBHSKfW-DCb740ccc5511e5e8fedcff06b081203-D7i6Ckgxk41i%2BdpejPaS5WTTeic%3D&to=65&size=1090604&sta_dx=1090604&sta_cs=1&sta_ft=mp3&sta_ct=0&sta_mt=0&fm2=MH%2CYangquan%2CAnywhere%2C%2CNone%2Cany&ctime=1572319467&mtime=1572319467&resv0=cdnback&resv1=0&resv2=rlim&resv3=5&resv4=1090604&vuk=2117437711&iv=0&htype=&randtype=&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-02895e0652184dc39c52b381582350dbfaf3590299ce4b74bd47909e9e7a0b470e39b0418692076aa08b5c8c0054e9c6c8abe51fcfd7e680305a5e1275657320&sl=68616270&expires=8h&rt=sh&r=602439256&vbdid=2939104332&fin=TTSOutput.mp3&fn=TTSOutput.mp3&rtype=1&dp-logid=6992234819097515635&dp-callid=0.1&hps=1&tsl=200&csl=200&csign=HMiavZmK%2B%2BVwV1XQ0oUsBZvJjhU%3D&so=0&ut=6&uter=4&serv=0&uc=738096701&ti=648eaef5c3fa81d1abb7b6c60806c5e1a5390508c7e5871f&reqlabel=250528_f&by=themis";
        audio.play();
        port.postMessage({answer: 'to speech succeed', status: 'ok'});
      });
  } else {
    throw Error('port name invalid');
  }
});
console.log('+++++++');
