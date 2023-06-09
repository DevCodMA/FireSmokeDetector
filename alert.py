from twilio.rest import Client
from time import time
from numpy import round

start  = time()

def generate(phone):
    global start
    # set Twilio account SID, auth token, and phone number
    account_sid = 'YOUR ACC SID'
    auth_token = 'AUTH TOKEN'
    twilio_number = 'TWILIO NUMBER'

    # create Twilio client
    client = Client(account_sid, auth_token)

    # send SMS message containing OTP
    end = time()

    if round(end-start, 0) > 300000:
        message = client.messages.create(
            body=f'Fire detected!!!',
            from_=twilio_number,
            to="+91"+phone
        )
        start = time()

    
