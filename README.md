# 📘 README

## 🚀 Instalación y Entorno Virtual

1. Clonar el repositorio (si corresponde) y entrar en la carpeta raíz:

   ```bash
   cd ppt-rag-groq
   ```

2. Crear un entorno virtual en Python 3:

   ```bash
   py -3 -m venv .venv
   ```

3. Activar el entorno virtual:

   - En **PowerShell (Windows)**:
     ```bash
     .\.venv\Scripts\Activate.ps1
     ```
   - En **CMD (Windows)**:
     ```bash
     .venv\Scripts\activate.bat
     ```
   - En **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```

4. Instalar dependencias necesarias:

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Servidor FastAPI con Uvicorn

Para iniciar la API local en `http://127.0.0.1:9000`:

```bash
uvicorn servidor:app --host 127.0.0.1 --port 9000 --workers 1
```

- `servidor:app` → indica el archivo `servidor.py` y la instancia `app` de FastAPI.  
- `--host 127.0.0.1` → solo accesible localmente.  
- `--port 9000` → expone en el puerto 9000.  
- `--workers 1` → cantidad de procesos del servidor.  

---

## 📑 Scripts Auxiliares

### 1. Ingesta de Embeddings

Este script genera los embeddings necesarios para la búsqueda semántica:

```bash
python scripts/ingesta_embeddings.py
```

### 2. Extracción de texto desde PPTX

Este script procesa presentaciones `.pptx` y extrae su contenido para indexarlo:

```bash
python scripts/extractor_pptx.py
```

---

## 🗂️ Estructura Recomendada

```
ppt-rag-groq/
├── servidor.py              # API principal (FastAPI)
├── enunciados.json          # Enunciados disponibles
├── requirements.txt         # Dependencias del proyecto
├── scripts/
│   ├── ingesta_embeddings.py
│   └── extractor_pptx.py
└── README.md
```

---

## ✅ Flujo de Trabajo

1. Crear/activar el entorno virtual.  
2. Instalar dependencias.  
3. Ejecutar los scripts necesarios (`ingesta_embeddings.py` y/o `extractor_pptx.py`).  
4. Levantar el servidor con Uvicorn.  
5. Probar los endpoints desde el navegador o con herramientas como **cURL**, **Postman** o el frontend conectado.  
