# 🚀 Portal 4 - Deployment & Production Guide

## 📋 Overview

Guía completa para despliegue y puesta en producción de Portal 4 AI Engineering Program, incluyendo configuración de infraestructura, CI/CD, monitoreo y mantenimiento.

---

## 🏗️ Architecture Overview

### Production Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION ENVIRONMENT                    │
├─────────────────────────────────────────────────────────────┤
│  Load Balancer (Cloudflare/AWS ALB)                       │
│  ├── CDN Cache (Static Assets)                             │
│  └── SSL Termination                                       │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ├── Docusaurus Site (Portal 4 Docs)                      │
│  ├── Interactive Jupyter Hub                               │
│  ├── API Gateway (FastAPI)                                 │
│  └── Authentication Service                                │
├─────────────────────────────────────────────────────────────┤
│  Services Layer                                             │
│  ├── User Progress Service                                  │
│  ├── Analytics Service                                     │
│  ├── Partnership API                                       │
│  └── Quality Dashboard                                     │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── PostgreSQL (User Data)                               │
│  ├── Redis (Cache & Sessions)                             │
│  ├── Vector DB (Content Embeddings)                       │
│  └── Object Storage (Assets)                              │
├─────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                │
│  ├── Prometheus (Metrics)                                  │
│  ├── Grafana (Dashboards)                                 │
│  ├── ELK Stack (Logging)                                  │
│  └── Sentry (Error Tracking)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🐳 Containerization Strategy

### Multi-Stage Dockerfile
```dockerfile
# =============================================================================
# Portal 4 - Production Dockerfile
# =============================================================================

# Stage 1: Build Environment
FROM node:18-alpine AS docs-builder
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .

# Build documentation site
RUN yarn build

# Stage 2: Python Environment
FROM python:3.11-slim AS python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# Stage 3: Production
FROM python-base AS production

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install -r requirements-prod.txt

# Copy built documentation
COPY --from=docs-builder /app/build ./docs/build

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "src.main:app"]
```

### Docker Compose - Development
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  portal4-app:
    build:
      context: .
      dockerfile: Dockerfile
      target: python-base
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/portal4_dev
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
    volumes:
      - .:/app
      - /app/node_modules
    depends_on:
      - db
      - redis
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

  docs:
    build:
      context: .
      dockerfile: Dockerfile.docs
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
    command: yarn start --host 0.0.0.0

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: portal4_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  jupyter:
    image: jupyter/datascience-notebook:latest
    ports:
      - "8888:8888"
    environment:
      - JUPYTER_ENABLE_LAB=yes
    volumes:
      - ./notebooks:/home/jovyan/work
      - jupyter_data:/home/jovyan

volumes:
  postgres_data:
  redis_data:
  jupyter_data:
```

### Docker Compose - Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  portal4-app:
    image: portal4/app:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - db
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - static_files:/app/static
    depends_on:
      - portal4-app
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  static_files:
```

---

## ☁️ Cloud Deployment Options

### Option 1: AWS ECS with Fargate
```yaml
# aws-ecs-task-definition.json
{
  "family": "portal4-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "portal4-app",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/portal4:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/portal4"
        },
        {
          "name": "REDIS_URL", 
          "value": "redis://elasticache-endpoint:6379"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:portal4-secrets"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/portal4",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### Option 2: Google Cloud Run
```yaml
# gcp-cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: portal4-service
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 3600
      containers:
      - image: gcr.io/PROJECT-ID/portal4:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: portal4-secrets
              key: database_url
        - name: REDIS_URL
          value: "redis://memorystore-ip:6379"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Option 3: Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: portal4-deployment
  labels:
    app: portal4
spec:
  replicas: 3
  selector:
    matchLabels:
      app: portal4
  template:
    metadata:
      labels:
        app: portal4
    spec:
      containers:
      - name: portal4-app
        image: portal4/app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: portal4-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: portal4-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: static-files
          mountPath: /app/static
      volumes:
      - name: static-files
        persistentVolumeClaim:
          claimName: portal4-static-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: portal4-service
spec:
  selector:
    app: portal4
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: portal4-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - portal4.ai
    - www.portal4.ai
    secretName: portal4-tls
  rules:
  - host: portal4.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: portal4-service
            port:
              number: 80
```

---

## 🔄 CI/CD Pipeline Enhanced

### GitHub Actions - Complete Production Pipeline
```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment Pipeline

on:
  push:
    branches: [main]
    tags: ['v*']
  
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  quality-gates:
    name: Quality Gates
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
        
    - name: Run quality checks
      run: |
        # Code formatting
        black --check src tests
        isort --check-only src tests
        
        # Linting
        flake8 src tests
        
        # Type checking
        mypy src
        
        # Security scanning
        bandit -r src
        safety check
        
        # Test coverage
        pytest tests/ --cov=src --cov-fail-under=85
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  build-and-push:
    name: Build and Push Container
    runs-on: ubuntu-latest
    needs: quality-gates
    permissions:
      contents: read
      packages: write
      
    outputs:
      image: ${{ steps.image.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=tag
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
          
    - name: Build and push image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        
    - name: Output image
      id: image
      run: |
        echo "image=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ steps.build.outputs.digest }}" >> $GITHUB_OUTPUT

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: build-and-push
    
    steps:
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ needs.build-and-push.outputs.image }}
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build-and-push, security-scan]
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
        
    - name: Update ECS service
      run: |
        aws ecs update-service \
          --cluster portal4-staging \
          --service portal4-staging-service \
          --task-definition portal4-staging:${{ github.run_number }} \
          --force-new-deployment
          
    - name: Wait for deployment
      run: |
        aws ecs wait services-stable \
          --cluster portal4-staging \
          --services portal4-staging-service
          
    - name: Run smoke tests
      run: |
        curl -f ${{ secrets.STAGING_URL }}/health
        pytest tests/smoke/ --url=${{ secrets.STAGING_URL }}

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
        
    - name: Blue-Green Deployment
      run: |
        # Create new task definition revision
        aws ecs register-task-definition \
          --cli-input-json file://aws-ecs-task-definition.json
          
        # Update service with new task definition
        aws ecs update-service \
          --cluster portal4-production \
          --service portal4-production-service \
          --task-definition portal4-production:${{ github.run_number }}
          
        # Wait for deployment to complete
        aws ecs wait services-stable \
          --cluster portal4-production \
          --services portal4-production-service
          
    - name: Health check
      run: |
        for i in {1..10}; do
          if curl -f ${{ secrets.PRODUCTION_URL }}/health; then
            echo "Health check passed"
            break
          fi
          echo "Health check failed, retrying in 30s..."
          sleep 30
        done
        
    - name: Run production tests
      run: |
        pytest tests/smoke/ --url=${{ secrets.PRODUCTION_URL }}
        
    - name: Notify success
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: "🚀 Production deployment successful! Portal 4 is live at ${{ secrets.PRODUCTION_URL }}"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

  rollback:
    name: Rollback on Failure
    runs-on: ubuntu-latest
    needs: deploy-production
    if: failure()
    environment: production
    
    steps:
    - name: Rollback to previous version
      run: |
        aws ecs update-service \
          --cluster portal4-production \
          --service portal4-production-service \
          --task-definition portal4-production:${{ github.run_number - 1 }}
```

---

## 📊 Monitoring & Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "portal4-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: 'portal4-app'
    static_configs:
      - targets: ['portal4-app:8000']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboard Config
```json
{
  "dashboard": {
    "id": null,
    "title": "Portal 4 - Production Dashboard",
    "tags": ["portal4", "production"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time P99",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99 Response Time"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "portal4_active_users",
            "legendFormat": "Active Users"
          }
        ]
      }
    ]
  }
}
```

---

## 🔧 Environment Configuration

### Production Environment Variables
```bash
# .env.production
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=portal4.ai,www.portal4.ai

# Database
DATABASE_URL=postgresql://user:pass@prod-db:5432/portal4_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis Cache
REDIS_URL=redis://prod-redis:6379/0
CACHE_TTL=3600

# External APIs
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_CLOUD_API_KEY=your-gcp-api-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_METRICS_ENABLED=true
LOG_LEVEL=INFO

# CDN & Assets
STATIC_URL=https://cdn.portal4.ai/static/
MEDIA_URL=https://cdn.portal4.ai/media/
AWS_S3_BUCKET=portal4-assets-prod

# Email
SMTP_HOST=smtp.mailgun.org
SMTP_PORT=587
SMTP_USER=your-smtp-user
SMTP_PASSWORD=your-smtp-password

# Security
CORS_ALLOWED_ORIGINS=https://portal4.ai,https://www.portal4.ai
CSRF_TRUSTED_ORIGINS=https://portal4.ai,https://www.portal4.ai
SSL_REDIRECT=true
SECURE_COOKIES=true
```

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] **Code Quality:** All tests passing (>85% coverage)
- [ ] **Security:** Vulnerability scans completed
- [ ] **Performance:** Load testing completed  
- [ ] **Database:** Migrations prepared and tested
- [ ] **Dependencies:** All production dependencies updated
- [ ] **Environment:** Production secrets configured
- [ ] **Monitoring:** Dashboards and alerts configured
- [ ] **Backup:** Database backup strategy in place

### Deployment
- [ ] **Blue-Green:** New version deployed to staging
- [ ] **Health Checks:** All endpoints responding correctly
- [ ] **Database Migration:** Schema updates applied
- [ ] **Cache Warmup:** Critical data pre-loaded
- [ ] **Traffic Switch:** Load balancer updated
- [ ] **Smoke Tests:** Core functionality verified
- [ ] **Monitoring:** Metrics and logs flowing correctly
- [ ] **Documentation:** Deployment notes updated

### Post-Deployment
- [ ] **Performance:** Response times within SLA
- [ ] **Error Rates:** Error rates below threshold (< 1%)
- [ ] **User Experience:** Key user journeys working
- [ ] **Analytics:** Tracking events firing correctly
- [ ] **Alerts:** No critical alerts triggered
- [ ] **Rollback Plan:** Tested and ready if needed
- [ ] **Team Notification:** Stakeholders informed
- [ ] **Post-Mortem:** Lessons learned documented

---

## 🎯 Success Metrics

### Technical KPIs
- **Uptime:** 99.9%+ availability
- **Performance:** <2s average response time
- **Error Rate:** <1% error rate
- **Security:** Zero security incidents
- **Deployment:** <15min deployment time

### Business KPIs  
- **User Engagement:** 80%+ weekly active users
- **Course Completion:** 75%+ completion rate
- **Job Placement:** 85%+ placement success
- **User Satisfaction:** 4.8/5 average rating
- **Cost Efficiency:** <$0.50 per user per month

**Portal 4 está listo para producción enterprise-grade con infraestructura escalable y monitoring comprehensivo.** 🚀
