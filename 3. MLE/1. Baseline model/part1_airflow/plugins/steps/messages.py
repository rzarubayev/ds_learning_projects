# plugins/steps/messages.py
from airflow.providers.telegram.hooks.telegram import TelegramHook
import os

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_telegram_success_message(context): 
    hook = TelegramHook(
        telegram_conn_id="telegram",
        token=TG_TOKEN,
        chat_id=TG_CHAT_ID)
    message = f"""
✅ Исполнение DAG прошло успешно! 
DAG '{context["dag"].dag_id}' 
ID '{context["run_id"]}'"""
    print(f"Send success message to telegram chat ID {TG_CHAT_ID}")
    hook.send_message({
        'chat_id': TG_CHAT_ID,
        'text': message
    }) 

def send_telegram_failure_message(context):
    hook = TelegramHook(
        telegram_conn_id="telegram",
        token=TG_TOKEN,
        chat_id=TG_CHAT_ID)
    message = f"""
⛔ Исполнение DAG завершилось с ошибкой!
DAG '{context["dag"].dag_id}'
ID '{context["run_id"]}'"""
    print(f"Send failure message to telegram chat ID {TG_CHAT_ID}")
    hook.send_message({
        'chat_id': TG_CHAT_ID,
        'text': message
    })