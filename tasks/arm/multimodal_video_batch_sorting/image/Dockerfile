FROM ghcr.io/agentslastexam/container-ubuntu22-base:latest

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission
ENV MPLBACKEND=Agg

RUN python3 -c "from PIL import Image; print('[image] Pillow import OK')" \
    && runuser -u user -- python3 -c "from PIL import Image; print('[image] agent-UID import OK')"
