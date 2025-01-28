# Flask 相關模組
from flask import Flask, request

# Line Bot 相關模組
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApiBlob,
)

# 圖片辨識相關模組
from ultralytics import YOLO
import cv2

# 其他模組
import requests
import configparser
import os
from datetime import datetime
from google.cloud import storage


# 固定儲存路徑
save_dir = os.path.join('tmp', 'static') 
os.makedirs(save_dir, exist_ok=True)
#當天日期，收容所的檔案名稱會用這個
current_date = str(datetime.now().date()) 
# 圖片需要提供的網址
base_url = os.getenv('base_url')


app = Flask(__name__, static_url_path="/static", static_folder="tmp/static") 
app.config['UPLOAD_FOLDER'] = 'tmp/static'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

#我將重要資訊裝進別的文件，不要寫進程式碼
config = configparser.ConfigParser() 
config.read("config.ini")
configuration = Configuration(access_token=config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))
line_login_id = config.get('line-bot', 'line_login_id')
line_login_secret = config.get('line-bot', 'line_login_secret')
HEADER = {
    'Content-type': 'application/json',
    'Authorization': F'Bearer {config.get("line-bot", "channel_access_token")}'
}



# Line的部分由POST傳、抓資料
@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return 'GET_ok'

    body = request.json
    events = body.get("events", [])

    if request.method == 'POST' and not events:
        return 'POST_ok'

    print(body)

    event = events[0]
    if "replyToken" not in event:
        return {}

    replyToken = event["replyToken"]
    payload = {"replyToken": replyToken}

    # 文字
    if event["type"] == "message" and event["message"]["type"] == "text":
        text = event["message"]["text"]
        if text == "@收容所":
            payload["messages"] = [shelter_zelda(), shelter_link()]
        elif text == "@新手飼養手冊":
            payload["messages"] = [manual()]
        else:
            payload["messages"] = [RAG(text)]
        replyMessage(payload)
        return payload

    # 圖片
    if event["type"] == "message" and event["message"]["type"] == "image":
        message_id = event["message"]["id"]
        payload["messages"] = [distinguish(message_id)]
        replyMessage(payload)
        return payload

    return {}



# n8n上的RAG
def RAG(text):
    # 測試用網址 --這邊需要更改
    # n8n_url = 'http://localhost:5678/webhook-test/'
    # 生產用網址
    n8n_url = 'http://localhost:5678/webhook/'

    data = {
        "text": text  # 將傳遞的文字作為參數傳送
    }
    # n8n處理後的訊息接收並回傳
    response = requests.post(n8n_url, json=data)

    # 確認返回的資料中是否包含 response 和 text 欄位
    try:
        n8n_response_data = response.json()
    
        RAG_text = n8n_response_data[0]["response"]["text"]
        
    except ValueError:
        # 如果回應不是有效的 JSON 格式
        print(f"無效的 JSON 格式: {response.text}")
        RAG_text = "勞動布出現，請稍後再試"

    message = {
        "type": "text",
        "text": RAG_text
    }
    return message

# 無情的辨識圖片開始
# 先抓圖片
def distinguish(message_id):
    """
    辨識圖片，由message_id向line官方要圖片
    """
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(message_id=message_id)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{message_id}.jpg")
        with open(image_path, 'wb') as f:
            f.write(message_content)

    # 確認圖片保存
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not saved correctly.")

    # 呼叫yolo模型，來得到辨識結果的url
    image_url = emotion(image_path, message_id)

    message = {
        "type": "image",
        "originalContentUrl": image_url,
        "previewImageUrl": image_url,
    }

    return message

# 分析圖片
def emotion(image_path, message_id):
    """
    分析圖片情緒
    Args:
        message_id是我想要命名的名稱
    Returns:
        url: 結果圖片url
    """

    # 使用 YOLO 模型辨識
    img = cv2.imread(image_path)
    model = YOLO('0108.pt')
    results = model(img)

    # 取最高可能性的
    if isinstance(results, list):
        results = results[0]

    # 保存 YOLO 辨識結果
    output_image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{message_id}_result.jpg")
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # 嘗試保存 YOLO 結果圖片
    try:
        results.save(output_image_path)
    except AttributeError:
        result_img = results.plot()
        cv2.imwrite(output_image_path, result_img)

    # 檢查結果圖片是否成功保存
    if not os.path.exists(output_image_path):
        raise FileNotFoundError(f"Output image file {output_image_path} was not saved correctly.")

    # 生成公開 url
    image_url = f"{base_url}/static/{message_id}_result.jpg"
    # 回傳圖片url
    return image_url


# 小蝸提供的google map地圖
def shelter_link():

    message = {
        "type": "text",
        "text": "想在google map上面查看? 前往連結：https://maps.app.goo.gl/DCfrDHKc17zV68tV6",
    }
    return message


# 我只是附帶的
def shelter_zelda():
    """
    根據當前日期生成圖片訊息字典。
    圖片檔案先下載到本地暫存目錄，再生成對應的圖片 URL。
    """
    
    image = f"{current_date}.jpg"
    image_path = os.path.join("/tmp", image)  # 暫存目錄

    # 檢查圖片是否存在，如果不存在才從GCS下載
    if not os.path.exists(image_path):
        download_to_tmp(f"shelter/image/{image}")

    image_url = f'{base_url}/static/{current_date}.jpg'

    # 返回圖片訊息字典
    message = {
        "type": "image",
        "originalContentUrl": image_url,
        "previewImageUrl": image_url,
    }
    return message


def download_to_tmp(source_blob_name):
    """
    從固定 GCS 儲存桶 ('orereo') 中下載指定檔案到本地暫存目錄。
    Args:
        source_blob_name (str): GCS 中的檔案路徑。
    Returns:
        str: 本地檔案下載路徑。
    """
    try:

        # 設定本地下載路徑，保留目錄結構
        local_path = os.path.join(save_dir, source_blob_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)  # 確保子目錄存在

        # 建立 GCS 客戶端
        storage_client = storage.Client()

        # 獲取 bucket 和 blob（檔案物件）
        bucket = storage_client.bucket('orereo')
        blob = bucket.blob(source_blob_name)

        # 將檔案下載到本地
        blob.download_to_filename(local_path)

        print(f"檔案已成功下載到：{local_path}")
        return local_path

    except Exception as e:
        print(f"檔案下載失敗：{e}")
        raise




def manual():

# 家瑋提供的文本內容    
    steps = [
        {
            "文本": "1研究不同狗的品種.jpg",
            "title": "研究不同狗的品種",
            "text": "step 1",
            "reply_text": "Information about different dog breeds",
            "action_text": "研究不同的犬種，了解他們的性格、需求和特點，以選擇適合您的生活方式的犬種。"
        },
        {
            "文本": "2犬舍.jpg",
            "title": "犬舍",
            "text": "step 2",
            "reply_text": "Information about dog breeds",
            "action_text": "尋找當地的動物收容所、犬舍或領養機構，以找到可供領養的狗。"
        },
        {
            "文本": "3填寫領養表單.jpg",
            "title": "填寫領養表單",
            "text": "step 3",
            "reply_text": "Information about adopting dog",
            "action_text": "填寫領養申請表，提供您的個人資料、家庭情況和居住環境等信息。"
        },
        {
            "文本": "4領養面談.jpg",
            "title": "領養面談",
            "text": "step 4",
            "reply_text": "Information about interview about adoption",
            "action_text": "領養機構的工作人員會與您進行面談，了解您的意圖、生活方式和對狗的照顧能力。"
        },
        {
            "文本": "5參觀狗狗.jpg",
            "title": "參觀狗狗",
            "text": "step 5",
            "reply_text": "Visit the dogs available for adoption and learn about their personalities and behaviors",
            "action_text": "您會有機會參觀可供領養的狗，了解牠們的性格和行為。"
        },
        {
            "文本": "6簽署合同.jpg",
            "title": "簽署合同",
            "text": "step 6",
            "reply_text": "If you decide to adopt a dog, you will complete a contract pledging to provide appropriate care and love.",
            "action_text": "如果您決定領養，將會完成相關的合約，承諾提供適當的照顧和愛護。"
        },
        {
            "文本": "7接狗回家.jpg",
            "title": "接狗回家",
            "text": "step 7",
            "reply_text": "adopt a dog to home",
            "action_text": "領養完成後，您可以接狗回家，開始新的生活。"
        }
    ]
# Line的格式
    columns = []
    for step in steps:
        columns.append({
            "thumbnailImageUrl": f"{base_url}/static/{step['文本']}",
            "imageBackgroundColor": "#FFFFFF",
            "title": step["title"],
            "text": step["text"],
            "defaultAction": {
                "type": "message",
                "label": "Reply with message",
                "text": step["reply_text"]
            },
            "actions": [
                {
                    "type": "message",
                    "label": "查看",
                    "text": step["action_text"]
                }
            ]
        })
    #將上面的column加入message
    message = {
        "type": "template",
        "altText": "this is a carousel template",
        "template": {
            "type": "carousel",
            "columns": columns,
            "imageAspectRatio": "rectangle",
            "imageSize": "cover"
        }
    }
    
    return message



# Line官方給的回覆訊息方法
def replyMessage(payload):
    url = "https://api.line.me/v2/bot/message/reply"
    response = requests.post(url=url, headers=HEADER, json=payload)
    if response.status_code == 200:
        return "reply_ok"
    else:
        print(response.text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
