FROM uehreka/flp-base:latest

RUN mkdir /root/FasterLivePortrait
COPY ./requirements.txt /root/FasterLivePortrait/requirements.txt
WORKDIR /root/FasterLivePortrait

RUN apt update && apt install python3-tk -y

RUN pip install -r requirements.txt

COPY ./src/models/XPose /root/FasterLivePortrait/src/models/XPose


# Setup frontend
# WORKDIR /root/FasterLivePortrait/frontend
# COPY ./frontend/package.json /root/FasterLivePortrait/frontend/package.json
# RUN npm install
# COPY ./frontend /root/FasterLivePortrait/frontend
# RUN npm run build
WORKDIR /root/FasterLivePortrait

COPY ./configs /root/FasterLivePortrait/configs
COPY ./scripts /root/FasterLivePortrait/scripts
COPY ./src /root/FasterLivePortrait/src
COPY ./assets /root/FasterLivePortrait/assets
COPY ./entrypoint.sh /root/FasterLivePortrait/entrypoint.sh
COPY ./websocket_server.py /root/FasterLivePortrait/websocket_server.py
COPY ./run.py /root/FasterLivePortrait/run.py

ENTRYPOINT [ "./entrypoint.sh" ]