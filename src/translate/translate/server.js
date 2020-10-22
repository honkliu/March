const http = require('http');
const express = require("express");
const app = express();
const xmlbuilder = require('xmlbuilder');
const rp = require('request-promise');
const fs = require('fs');
// var router = express.Router();
const bodyParser = require('body-parser');
var querystring = require('querystring');
const audioPath = 'TTSOutput.mp3';
// Gets an access token.
function getAccessToken(subscriptionKey) {
    let options = {
        method: 'POST',
        uri: 'https://southeastasia.api.cognitive.microsoft.com/sts/v1.0/issuetoken',
        // uri: 'https://westus.api.cognitive.microsoft.com/sts/v1.0/issueToken',
        headers: {
            'Ocp-Apim-Subscription-Key': subscriptionKey,
            'Content-type' : 'application/x-www-form-urlencoded',
        }
    }
    return rp(options);
}

// Converts text to speech using the input from readline.
function textToSpeech(accessToken, text) {
    // Create the SSML request.
    let xml_body = xmlbuilder.create('speak')
    .att('version', '1.0')
    .att('xml:lang', 'en-us')
    .ele('voice')
    .att('xml:lang', 'en-us')
    // .att('name', 'en-US-Guy24kRUS') // Short name for 'Microsoft Server Speech Text to Speech Voice (en-US, Guy24KRUS)'
    .att('name', 'en-US-JessaNeural') // Short name for 'Microsoft Server Speech Text to Speech Voice (en-US, Guy24KRUS)'
    .txt(text)
    .end();
    // Convert the XML into a string to send in the TTS request.
    let body = xml_body.toString();
    
    let options = {
			method: 'POST',
			baseUrl: 'https://southeastasia.tts.speech.microsoft.com/',
			// baseUrl: 'https://westus.tts.speech.microsoft.com/',
			url: 'cognitiveservices/v1',
			headers: {
				'Authorization': 'Bearer ' + accessToken,
				// 'Ocp-Apim-Subscription-Key': 'ea80d576254f4df6860cc730a1981358',
				'cache-control': 'no-cache',
				'User-Agent': 'YOUR_RESOURCE_NAME',
				'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
				'Content-Type': 'application/ssml+xml'
			},
			body: body
    }
    
    let request = rp(options)
    .on('response', (response) => {
			if (response.statusCode === 200) {
				request.pipe(fs.createWriteStream(audioPath))
				.on('error', function(e){  console.error(e)})
				console.log('\nYour file is ready.\n')
			}
    });
    return request;

};

async function main(req, res) {
    const subscriptionKey = 'ea80d576254f4df6860cc730a1981358';
    const text = req.query.text;
    if (!subscriptionKey) {
        throw new Error('Environment variable for your subscription key is not set.')
    };
    try {
			const accessToken = await getAccessToken(subscriptionKey);
			await textToSpeech(accessToken, text);
			res.writeHead(200, {
							'Content-Type': 'text/plain;charset=utf-8','Access-Control-Allow-Origin':'*'
			});
			// res.end(`audioPath: ${audioPath}`);
			res.end(audioPath);
    } catch (err) {
      console.log(`Something went wrong: ${err}`);
    }
}
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.post('/texttospeech', main);
 
app.listen(2000)

