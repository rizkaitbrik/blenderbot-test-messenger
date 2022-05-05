from flask import Flask, request
from Keys import facebook_messenger_access_token
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
from pymessenger import Bot

bot = Bot(facebook_messenger_access_token)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
mname = "facebook/blenderbot-1B-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
model = model.to(device)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)


def process_message(message):
    # get the message
    utterance = "<s> " + message + " </s>"
    inputs = tokenizer([utterance], return_tensors="pt").to(device)
    reply_ids = model.generate(**inputs)
    reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return reply


app = Flask(__name__)

VERIFY_TOKEN = "123456789"
PAGE_ACCESS_TOKEN = facebook_messenger_access_token


@app.route('/', methods=['POST', 'GET'])
def webhook():  # put application's code here
    if request.method == 'GET':
        if request.args.get('hub.verify_token') == VERIFY_TOKEN:
            return request.args.get('hub.challenge')
        else:
            return "Error, unable to connect to Facebook"
    elif request.method == 'POST':
        payload = request.json
        payload_entry = payload['entry']
        for entry in payload_entry:
            for message in entry['messaging']:
                if 'message' in message:
                    text = message['message']['text']
                    sender_id = message['sender']['id']
                    bot.send_text_message(sender_id, process_message(text))
        return "ok"
    else:
        return "200 OK"


if __name__ == '__main__':
    app.run()
