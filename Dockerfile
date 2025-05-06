FROM uehreka/flp-base:latest

USER root
RUN apt-get update && \
    apt-get install -y python3-tk libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/FasterLivePortrait/spd_editor
COPY ./spd_editor/requirements.txt /root/FasterLivePortrait/spd_editor/requirements.txt
WORKDIR /root/FasterLivePortrait/spd_editor
RUN pip install -r requirements.txt

COPY ./requirements.txt /root/FasterLivePortrait/requirements.txt
WORKDIR /root/FasterLivePortrait

RUN pip install -r requirements.txt

# Setup frontend
# WORKDIR /root/FasterLivePortrait/frontend
# COPY ./frontend/package.json /root/FasterLivePortrait/frontend/package.json
# RUN npm install
# COPY ./frontend /root/FasterLivePortrait/frontend
# RUN npm run build
WORKDIR /root/FasterLivePortrait

COPY ./spd_editor /root/FasterLivePortrait/spd_editor
COPY ./configs /root/FasterLivePortrait/configs
COPY ./scripts /root/FasterLivePortrait/scripts
COPY ./src /root/FasterLivePortrait/src
COPY ./assets /root/FasterLivePortrait/assets
COPY ./entrypoint.sh /root/FasterLivePortrait/entrypoint.sh
COPY ./websocket_server.py /root/FasterLivePortrait/websocket_server.py
COPY ./run.py /root/FasterLivePortrait/run.py

ENTRYPOINT [ "./entrypoint.sh" ]