FROM uehreka/flp-base:latest


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    python3-pil \
    python3-pil.imagetk \
    libx11-6 \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    libfreetype6-dev \
    libfontconfig1 \
    xauth \
    x11-apps \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /root/FasterLivePortrait
COPY ./requirements.txt /root/FasterLivePortrait/requirements.txt
WORKDIR /root/FasterLivePortrait

RUN pip install -r requirements.txt

COPY ./spd_editor/requirements.txt /root/FasterLivePortrait/spd_editor/requirements.txt
RUN cd spd_editor \
    && pip install -r requirements.txt

# Setup frontend
WORKDIR /root/FasterLivePortrait/frontend
COPY ./frontend/package.json /root/FasterLivePortrait/frontend/package.json
RUN npm install
COPY ./frontend /root/FasterLivePortrait/frontend
RUN npm run build
WORKDIR /root/FasterLivePortrait


RUN apt-get update && apt-get install -y --no-install-recommends \
libxcb-xinerama0 \
libxcb-icccm4 \
libxcb-image0 \
libxcb-keysyms1 \
libxcb-render-util0 \
libxcb-randr0 \
libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./spd_editor /root/FasterLivePortrait/spd_editor
COPY ./configs /root/FasterLivePortrait/configs
COPY ./scripts /root/FasterLivePortrait/scripts
COPY ./src /root/FasterLivePortrait/src
COPY ./assets /root/FasterLivePortrait/assets
COPY ./entrypoint.sh /root/FasterLivePortrait/entrypoint.sh
COPY ./websocket_server.py /root/FasterLivePortrait/websocket_server.py
COPY ./run.py /root/FasterLivePortrait/run.py
COPY ./app.py /root/FasterLivePortrait/app.py
COPY ./preprocess.py /root/FasterLivePortrait/preprocess.py

ENTRYPOINT [ "./entrypoint.sh" ]