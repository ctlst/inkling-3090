FROM vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
COPY patches/ /opt/inkling-3090/patches/
RUN cd /usr/local/lib/python3.14/dist-packages/vllm/models/inkling2 && \
    patch --batch --forward --fuzz=0 -p1 < /opt/inkling-3090/patches/inkling-sm86.patch && \
    sha256sum --check /opt/inkling-3090/patches/model-code.sha256 && \
    python3 -m compileall -q -f .
