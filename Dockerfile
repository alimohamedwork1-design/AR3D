FROM node:22-slim AS node
FROM runpod/flash@sha256:039efd6638ec6d6728246bb5130d8976c2f39be357c4ddd1e71866b238474848
WORKDIR /workspace
ARG ARQARY_BUILD_SHA=unknown
ENV ARQARY_WORKER_BUILD_SHA=${ARQARY_BUILD_SHA}
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=4
ENV OMP_NUM_THREADS=4
ENV COLMAP_PYTHON_ENGINE=1
ENV GS_ITERATIONS_DEFAULT=3000
COPY --from=node /usr/local/bin/ /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
RUN npm install -g @playcanvas/splat-transform@3.4.2
RUN pip install --no-cache-dir gsplat==1.5.3 pycolmap==4.2.1 ninja==1.13.2 \
    jaxtyping==0.3.9 scipy==1.18.1 pillow==13.0.0 requests==2.34.2 \
    plyfile==1.1.5 pygltflib==1.16.5
COPY handler.py asset_transfer.py private_scan.py gsplat_backend.py gsplat_smoke.py photogrammetry_fixture.py reconstruction_worker.py LICENSE.gsplat /workspace/
CMD ["python", "-u", "/workspace/reconstruction_worker.py"]
