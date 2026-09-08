# Source - https://stackoverflow.com/a/75118239
# Posted by Yasantha Biyuranga, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-08, License - CC BY-SA 4.0

import requests
from telegram_token import TOKEN  # Ensure you have a telegram_token.py file with your bot token


def send_telegram_message(message: str):
	chat_id = "6008334570"
	url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
	response = requests.get(url)
	if response.status_code != 200:
		print(f"Failed to send message: {response.status_code}, {response.text}")