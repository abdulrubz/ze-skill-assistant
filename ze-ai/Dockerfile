# The service image. Specs, generated tools and domain/ are baked in: tools
# change through a reviewed spec-sync PR and a redeploy, never at runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/ze-ai

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY domain ./domain
COPY specs ./specs
COPY generated ./generated
COPY specs.lock.json ./
COPY scripts ./scripts

RUN useradd --create-home ze && mkdir -p var && chown ze var
USER ze
EXPOSE 8000
# All configuration comes from the environment (see .env.example).
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
