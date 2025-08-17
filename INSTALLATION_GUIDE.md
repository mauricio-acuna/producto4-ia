# 🛠️ Portal 4 - Installation & Setup Guide

## 📋 Quick Start

Portal 4 puede ser instalado y configurado en múltiples formas dependiendo de tu necesidad:

### 🚀 Para Estudiantes (Uso Personal)
- **Opción 1:** [Local Development Setup](#local-development)
- **Opción 2:** [Docker Compose](#docker-setup)
- **Opción 3:** [GitHub Codespaces](#codespaces-setup)

### 🏢 Para Instituciones (Production)
- **Opción 1:** [Cloud Deployment (AWS/GCP/Azure)](#cloud-deployment)
- **Opción 2:** [Kubernetes Cluster](#kubernetes-setup)  
- **Opción 3:** [On-Premise Installation](#on-premise-setup)

---

## 💻 Local Development Setup

### Prerrequisitos
```bash
# Verificar versiones mínimas
python --version    # >= 3.9
node --version      # >= 18.0
git --version       # >= 2.30
docker --version    # >= 20.10 (opcional)
```

### Paso 1: Clonar Repositorio
```bash
# Clonar el repositorio
git clone https://github.com/mauricio-acuna/producto4-ia.git
cd producto4-ia

# Verificar estructura
ls -la
```

### Paso 2: Setup Python Environment
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Activar entorno (Windows)
venv\Scripts\activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias básicas
pip install -r requirements.txt

# Instalar dependencias de desarrollo (opcional)
pip install -r requirements-test.txt
```

### Paso 3: Configurar Variables de Entorno
```bash
# Copiar template de configuración
cp .env.example .env

# Editar configuración
nano .env  # o tu editor preferido
```

#### Contenido de .env (mínimo)
```bash
# Aplicación
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production

# Base de datos (SQLite para desarrollo local)
DATABASE_URL=sqlite:///./portal4_dev.db

# APIs (opcional para features avanzadas)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Logging
LOG_LEVEL=DEBUG
```

### Paso 4: Inicializar Base de Datos
```bash
# Crear estructura de base de datos
python scripts/init_database.py

# Cargar datos de ejemplo (opcional)
python scripts/load_sample_data.py
```

### Paso 5: Ejecutar Tests
```bash
# Verificar que todo funciona
pytest tests/ -v

# Tests con coverage
pytest tests/ --cov=src --cov-report=html

# Solo tests básicos (más rápido)
pytest tests/unit/ -v
```

### Paso 6: Iniciar Aplicación
```bash
# Modo desarrollo con auto-reload
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# La aplicación estará disponible en:
# http://localhost:8000
```

### Paso 7: Acceder a Documentación
```bash
# En una nueva terminal, iniciar documentación
cd docs/
npm install
npm start

# Documentación disponible en:
# http://localhost:3000
```

---

## 🐳 Docker Setup

### Opción A: Docker Compose (Recomendado)
```bash
# Clonar repositorio
git clone https://github.com/mauricio-acuna/producto4-ia.git
cd producto4-ia

# Configurar environment
cp .env.example .env
# Editar .env con tus configuraciones

# Iniciar todos los servicios
docker-compose up -d

# Verificar que todos los servicios están running
docker-compose ps

# Ver logs
docker-compose logs -f portal4-app

# Acceder a la aplicación
open http://localhost:8000
```

### Opción B: Docker Individual
```bash
# Build imagen
docker build -t portal4:latest .

# Ejecutar contenedor
docker run -d \
  --name portal4-app \
  -p 8000:8000 \
  -e DATABASE_URL=sqlite:///./data/portal4.db \
  -v $(pwd)/data:/app/data \
  portal4:latest

# Ver logs
docker logs -f portal4-app
```

### Docker Compose Services Incluidos
```yaml
Services disponibles:
- portal4-app:8000    # Aplicación principal
- docs:3000           # Documentación Docusaurus  
- postgres:5432       # Base de datos PostgreSQL
- redis:6379          # Cache y sessions
- jupyter:8888        # Notebooks interactivos
- grafana:3001        # Monitoring dashboard
```

---

## ☁️ GitHub Codespaces Setup

### Opción más Rápida - Zero Setup
```bash
# 1. Ir a: https://github.com/mauricio-acuna/producto4-ia
# 2. Click en "Code" > "Codespaces" > "Create codespace"
# 3. Esperar ~2-3 minutos para setup automático
# 4. Portal 4 estará disponible automáticamente
```

### Post-Setup en Codespaces
```bash
# El codespace incluye automáticamente:
✅ Python 3.11 environment
✅ Node.js 18 environment  
✅ Docker disponible
✅ Todas las dependencias instaladas
✅ Base de datos inicializada
✅ Tests ejecutados

# Comandos disponibles:
make dev        # Iniciar desarrollo
make test       # Ejecutar tests
make docs       # Iniciar documentación
make quality    # Verificar código
```

---

## 🏢 Cloud Deployment

### AWS ECS Deployment
```bash
# 1. Instalar AWS CLI
pip install awscli

# 2. Configurar credenciales
aws configure

# 3. Deploy usando CloudFormation
aws cloudformation deploy \
  --template-file aws-infrastructure.yml \
  --stack-name portal4-production \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    Environment=production \
    DatabasePassword=your-secure-password

# 4. Verificar deployment
aws ecs describe-services \
  --cluster portal4-production \
  --services portal4-service
```

### Google Cloud Run Deployment
```bash
# 1. Instalar gcloud CLI
curl https://sdk.cloud.google.com | bash

# 2. Configurar proyecto
gcloud config set project your-project-id
gcloud auth login

# 3. Build y deploy
gcloud builds submit --tag gcr.io/your-project-id/portal4
gcloud run deploy portal4 \
  --image gcr.io/your-project-id/portal4 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances
```bash
# 1. Instalar Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 2. Login
az login

# 3. Crear resource group
az group create --name portal4-rg --location eastus

# 4. Deploy container
az container create \
  --resource-group portal4-rg \
  --name portal4-app \
  --image portal4/app:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label portal4-demo
```

---

## ⚙️ Kubernetes Setup

### Minikube (Local Kubernetes)
```bash
# 1. Instalar minikube
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube /usr/local/bin/

# 2. Iniciar cluster
minikube start --cpus=4 --memory=8g

# 3. Aplicar manifests
kubectl apply -f k8s/

# 4. Acceder a la aplicación
minikube service portal4-service --url
```

### Producción Kubernetes
```bash
# 1. Configurar kubectl con tu cluster
kubectl config use-context your-production-cluster

# 2. Crear namespace
kubectl create namespace portal4

# 3. Aplicar secrets
kubectl apply -f k8s/secrets/ -n portal4

# 4. Aplicar configuraciones
kubectl apply -f k8s/configs/ -n portal4

# 5. Deploy aplicación
kubectl apply -f k8s/deployments/ -n portal4

# 6. Verificar deployment
kubectl get pods -n portal4
kubectl get services -n portal4
```

---

## 🏠 On-Premise Setup

### Requisitos de Hardware
```yaml
Minimum Requirements:
  CPU: 4 cores
  RAM: 8 GB
  Storage: 50 GB SSD
  Network: 100 Mbps

Recommended (Production):
  CPU: 8 cores
  RAM: 16 GB  
  Storage: 200 GB SSD
  Network: 1 Gbps
  Backup: External storage
```

### Ubuntu Server Setup
```bash
# 1. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Instalar dependencias
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  nodejs \
  npm \
  postgresql \
  redis-server \
  nginx \
  certbot \
  python3-certbot-nginx

# 3. Configurar PostgreSQL
sudo -u postgres createuser portal4
sudo -u postgres createdb portal4_prod
sudo -u postgres psql -c "ALTER USER portal4 PASSWORD 'secure-password';"

# 4. Clonar y configurar aplicación
git clone https://github.com/mauricio-acuna/producto4-ia.git /opt/portal4
cd /opt/portal4

# 5. Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Configurar variables de entorno
sudo cp .env.production /opt/portal4/.env
sudo chown portal4:portal4 /opt/portal4/.env

# 7. Configurar systemd service
sudo cp scripts/portal4.service /etc/systemd/system/
sudo systemctl enable portal4
sudo systemctl start portal4

# 8. Configurar nginx
sudo cp scripts/nginx.conf /etc/nginx/sites-available/portal4
sudo ln -s /etc/nginx/sites-available/portal4 /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# 9. Configurar SSL
sudo certbot --nginx -d portal4.yourdomain.com
```

---

## 🔧 Troubleshooting Common Issues

### Issue 1: Puerto 8000 en uso
```bash
# Encontrar proceso usando el puerto
lsof -i :8000

# Matar proceso
kill -9 <PID>

# O usar puerto diferente
uvicorn src.main:app --port 8001
```

### Issue 2: Dependencias no se instalan
```bash
# Actualizar pip y setuptools
pip install --upgrade pip setuptools wheel

# Limpiar cache
pip cache purge

# Instalar con verbose para debug
pip install -r requirements.txt -v
```

### Issue 3: Base de datos no conecta
```bash
# Verificar PostgreSQL está running
sudo systemctl status postgresql

# Verificar usuario y permisos
sudo -u postgres psql -c "\du"

# Test conexión
python -c "import psycopg2; psycopg2.connect('postgresql://user:pass@localhost:5432/portal4')"
```

### Issue 4: Docker issues
```bash
# Limpiar containers y imágenes
docker system prune -a

# Rebuild sin cache
docker-compose build --no-cache

# Verificar logs
docker-compose logs portal4-app
```

### Issue 5: Tests fallan
```bash
# Ejecutar tests en modo verbose
pytest tests/ -v -s

# Ejecutar solo tests que fallan
pytest tests/ --lf

# Verificar environment variables
python -c "import os; print(os.environ.get('DATABASE_URL'))"
```

---

## 📚 Next Steps

### Para Estudiantes
1. **Explorar Módulos:** Comenzar con [Módulo A - Fundamentos](./modulos/modulo-a-fundamentos/)
2. **Setup Development Environment:** Seguir [local development setup](#local-development)
3. **Unirse a Comunidad:** Discord/Slack communities
4. **Primer Proyecto:** Empezar con el copiloto de desarrollo

### Para Instructores
1. **Institution Setup:** Configurar [cloud deployment](#cloud-deployment)
2. **Student Onboarding:** Preparar [codespaces setup](#codespaces-setup)
3. **Custom Configuration:** Adaptar contenido según necesidades
4. **Monitoring Setup:** Configurar dashboards y alertas

### Para Empresas
1. **Enterprise Deployment:** [Kubernetes production setup](#kubernetes-setup)
2. **Integration Planning:** APIs y SSO integration
3. **Customization:** Adaptar currículo a necesidades empresariales
4. **Support Contract:** Contactar para enterprise support

---

## 📞 Support & Community

### 🆘 Getting Help
- **Documentation:** [Portal 4 Docs](https://portal4.ai/docs)
- **GitHub Issues:** [Report bugs/features](https://github.com/mauricio-acuna/producto4-ia/issues)
- **Discord Community:** [Join Portal 4 Discord](https://discord.gg/portal4)
- **Email Support:** support@portal4.ai

### 🤝 Contributing
- **Code Contributions:** Ver [CONTRIBUTING.md](./CONTRIBUTING.md)
- **Documentation:** Mejorar docs existente
- **Community Support:** Ayudar a otros students
- **Bug Reports:** Reportar issues con detalles

**¡Portal 4 está listo para transformar tu carrera en AI Engineering!** 🚀
