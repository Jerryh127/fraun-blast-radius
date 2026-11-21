# The scorer builds this on linux/amd64 and runs it with no network. The digest
# is the image the organisers labeled every commit in: this entry runs sqlglot's
# tests while it decides, so the Python and SQLite here must be theirs.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# One thread per numeric library: the run is limited to 256 processes and pays for every CPU second.
ENV PYTHONUNBUFFERED=1 SKIP_INTEGRATION=0 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# What sqlglot's tests import, at the versions in requirements.lock.
RUN pip install --no-cache-dir \
    duckdb==2.0.0 numpy==2.5.5 pandas==3.1.1 python-dateutil==4.0.0 pytz==2026.6 tzdata==2026.6

# sqlglot at the pinned commit.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /opt/sqlglot \
    && tar -xzf /tmp/sqlglot.tar.gz -C /opt/sqlglot --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz

COPY blast /app/blast
COPY selector.py /app/selector.py

# Compile once, then learn from the pinned commit which lines, classes and
# fixtures every test touches. Takes a few minutes; nothing here sees a patch.
RUN python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests /app \
    && cd /app \
    && python -m blast.build_map /opt/sqlglot /opt/blast-map --jobs 4 \
    && chmod -R a+rX /opt/sqlglot /opt/blast-map /app

CMD ["python", "/app/selector.py"]
